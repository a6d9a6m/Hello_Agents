"""实验性记忆子系统的协调器。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.memory.base import BaseMemory, MemoryConfig, MemoryItem, MemoryType
from src.memory.embedding import create_embedding_service
from src.memory.types.episodic import EpisodicMemory
from src.memory.types.perceptual import PerceptualMemory
from src.memory.types.semantic import SemanticMemory
from src.memory.types.working import WorkingMemory


class MemoryManager:
    """协调不同记忆后端的管理器。"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        """初始化记忆管理器。
        
        参数:
            config: 记忆配置，默认为从环境变量读取的配置
        """
        self.config = config or MemoryConfig.from_env()
        self.embedding_service = create_embedding_service(self.config)

        self.memories: Dict[MemoryType, BaseMemory] = {
            MemoryType.WORKING: WorkingMemory(self.config),
            MemoryType.EPISODIC: EpisodicMemory(self.config, self.embedding_service),
            MemoryType.SEMANTIC: SemanticMemory(self.config, self.embedding_service),
            MemoryType.PERCEPTUAL: PerceptualMemory(self.config, self.embedding_service),
        }

    def store(self, content: str, memory_type: MemoryType, **metadata) -> str:
        """存储记忆内容。
        
        参数:
            content: 记忆内容文本
            memory_type: 记忆类型
            **metadata: 额外元数据
            
        返回:
            存储的记忆项ID
        """
        # manager 只负责尽力补齐公共 embedding，不能反过来破坏各记忆类型自己的降级能力。
        # 例如工作记忆本就支持本地 TF-IDF，情景/语义/感知记忆也各自实现了无外部服务时的回退路径。
        # 因此这里一旦嵌入服务不可用，就保留 `None`，交给具体 memory 后端决定如何处理。
        embedding = None
        if self.embedding_service is not None and content.strip():
            try:
                embedding = self.embedding_service.embed(content)
            except Exception:
                embedding = None

        importance = metadata.pop("importance", 0.5)
        emotion = metadata.pop("emotion", None)
        tags = metadata.pop("tags", [])
        location = metadata.pop("location", None)
        source = metadata.pop("source", None)

        item = MemoryItem(
            id=self._generate_id(),
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            metadata=metadata,
            importance=importance,
            emotion=emotion,
            tags=tags,
            location=location,
            source=source,
        )

        memory_id = self.memories[memory_type].store(item)

        if memory_type == MemoryType.EPISODIC:
            self._link_episodic_to_semantic(item)

        return memory_id

    def retrieve(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
    ) -> List[MemoryItem]:
        """检索记忆内容。
        
        参数:
            query: 查询文本
            memory_types: 要检索的记忆类型列表，默认为所有类型
            limit: 返回结果的最大数量
            
        返回:
            按相关性排序的记忆项列表
        """
        if memory_types is None:
            memory_types = list(MemoryType)

        results: List[MemoryItem] = []
        for memory_type in memory_types:
            memory = self.memories[memory_type]
            items = memory.retrieve(query, limit=limit)
            results.extend(items)

        return self._sort_results(query=query, results=results, limit=limit)

    def _sort_results(self, query: str, results: List[MemoryItem], limit: int) -> List[MemoryItem]:
        """统一排序检索结果。

        说明:
            - 若某个记忆后端提供内部评分，则优先使用后端评分。
            - 否则继续沿用现有的 embedding 余弦排序逻辑。
            - 这样可以让工作记忆使用自身的时间衰减与重要性公式，同时不影响其他记忆类型。
        """
        query_embedding = self.embedding_service.embed(query)
        results.sort(
            key=lambda item: self._score_result_item(
                query=query,
                query_embedding=query_embedding,
                item=item,
            ),
            reverse=True,
        )

        return results[:limit]

    def _score_result_item(
        self,
        query: str,
        query_embedding: List[float],
        item: MemoryItem,
    ) -> float:
        """为单条结果计算排序分数。"""
        memory = self.memories[item.memory_type]
        custom_score = memory.score_item(query, item)
        if custom_score is not None:
            return custom_score

        return self.embedding_service.cosine_similarity(
            query_embedding,
            item.embedding or [],
        )

    def retrieve_context(self, query: str, context_size: int = 5) -> str:
        """检索上下文信息。
        
        参数:
            query: 查询文本
            context_size: 上下文大小
            
        返回:
            格式化后的上下文字符串
        """
        memories = self.retrieve(query, limit=context_size)
        context_parts = [f"[{memory.memory_type.value}] {memory.content}" for memory in memories]
        return "\n".join(context_parts)

    def delete(self, memory_id: str, memory_type: MemoryType) -> bool:
        """删除记忆项。
        
        参数:
            memory_id: 要删除的记忆项ID
            memory_type: 记忆类型
            
        返回:
            是否成功删除
        """
        return self.memories[memory_type].delete(memory_id)

    def update(self, memory_id: str, memory_type: MemoryType, updates: Dict[str, Any]) -> bool:
        """更新记忆项。
        
        参数:
            memory_id: 要更新的记忆项ID
            memory_type: 记忆类型
            updates: 更新字段的字典
            
        返回:
            是否成功更新
        """
        return self.memories[memory_type].update(memory_id, updates)

    def clear(self, memory_type: Optional[MemoryType] = None) -> bool:
        """清除记忆。
        
        参数:
            memory_type: 要清除的记忆类型，默认为清除所有类型
            
        返回:
            是否成功清除
        """
        if memory_type:
            return self.memories[memory_type].clear()

        success = True
        for memory in self.memories.values():
            success = success and memory.clear()
        return success

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息。
        
        返回:
            包含各类统计信息的字典
        """
        stats: Dict[str, Any] = {}
        total_count = 0
        total_importance = 0.0

        for memory_type, memory in self.memories.items():
            items = memory.get_all_items()
            count = len(items)
            total_count += count

            if count > 0:
                avg_importance = sum(item.importance for item in items) / count
                total_importance += avg_importance * count
            else:
                avg_importance = 0.0

            emotion_stats: Dict[str, int] = {}
            for item in items:
                if item.emotion:
                    emotion_stats[item.emotion] = emotion_stats.get(item.emotion, 0) + 1

            stats[memory_type.value] = {
                "count": count,
                "avg_importance": round(avg_importance, 3),
                "emotion_distribution": emotion_stats,
                "recent_count": len(
                    [item for item in items if (datetime.now() - item.created_at).days < 7]
                ),
                "last_updated": datetime.now().isoformat(),
            }

        overall_avg_importance = total_importance / total_count if total_count > 0 else 0
        stats["overall"] = {
            "total_count": total_count,
            "avg_importance": round(overall_avg_importance, 3),
            "memory_types": len(MemoryType),
        }

        return stats

    def _generate_id(self) -> str:
        """生成唯一标识符。
        
        返回:
            UUID字符串
        """
        import uuid

        return str(uuid.uuid4())

    def search(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        search_mode: str = "hybrid",
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[MemoryItem]:
        """搜索记忆内容。
        
        参数:
            query: 查询文本
            memory_types: 要搜索的记忆类型列表，默认为所有类型
            search_mode: 搜索模式（keyword/semantic/hybrid）
            limit: 返回结果的最大数量
            min_importance: 最小重要性阈值
            
        返回:
            按相关性排序的搜索结果列表
        """
        if memory_types is None:
            memory_types = list(MemoryType)

        all_results: List[MemoryItem] = []

        for memory_type in memory_types:
            memory = self.memories[memory_type]

            if search_mode == "keyword":
                items = memory.search_by_keyword(query, limit=limit)
            elif search_mode == "semantic":
                items = memory.search_by_semantic(query, limit=limit)
            elif search_mode == "hybrid":
                items = memory.search_hybrid(query, limit=limit)
            else:
                items = memory.retrieve(query, limit=limit)

            items = [item for item in items if item.importance >= min_importance]
            all_results.extend(items)

        return self._sort_results(query=query, results=all_results, limit=limit)

    def store_perceptual(
        self,
        content: str = "",
        *,
        media_type: str = "text",
        media_data: Optional[Any] = None,
        **metadata,
    ) -> str:
        """通过 manager 存储感知记忆。

        说明：
            - 文本、图像、音频等入口统一收敛到 PerceptualMemory。
            - manager 负责补齐公共字段，具体模态隔离和桥接逻辑由感知记忆实现。
        """
        importance = float(metadata.pop("importance", 0.5))
        emotion = metadata.pop("emotion", None)
        tags = metadata.pop("tags", [])
        location = metadata.pop("location", None)
        source = metadata.pop("source", None)

        item = MemoryItem(
            id=self._generate_id(),
            content=content,
            memory_type=MemoryType.PERCEPTUAL,
            embedding=None,
            metadata=metadata,
            importance=importance,
            emotion=emotion,
            tags=tags,
            location=location,
            source=source,
        )
        perceptual_memory = self.memories[MemoryType.PERCEPTUAL]
        return perceptual_memory.store(item, media_data=media_data, media_type=media_type)

    def search_perceptual(
        self,
        query: str,
        *,
        search_mode: str = "hybrid",
        query_media_type: str = "text",
        target_media_types: Optional[List[str]] = None,
        cross_modal: bool = True,
        limit: int = 10,
    ) -> List[MemoryItem]:
        """通过 manager 执行感知记忆检索。"""
        perceptual_memory = self.memories[MemoryType.PERCEPTUAL]

        if search_mode == "keyword":
            return perceptual_memory.search_by_keyword(
                query,
                limit=limit,
                query_media_type=query_media_type,
                target_media_types=target_media_types,
                cross_modal=cross_modal,
            )
        if search_mode == "semantic":
            return perceptual_memory.search_by_semantic(
                query,
                limit=limit,
                query_media_type=query_media_type,
                target_media_types=target_media_types,
                cross_modal=cross_modal,
            )
        if search_mode == "all":
            return perceptual_memory.get_all_items(target_media_types=target_media_types)[:limit]

        return perceptual_memory.search_hybrid(
            query,
            limit=limit,
            query_media_type=query_media_type,
            target_media_types=target_media_types,
            cross_modal=cross_modal,
        )

    def forget(self, strategy: str = "importance_based", **kwargs) -> Dict[str, Any]:
        """遗忘记忆（删除低价值记忆）。
        
        参数:
            strategy: 遗忘策略
            **kwargs: 策略相关参数
            
        返回:
            包含删除统计信息的字典
        """
        deleted_count = 0
        deleted_items: List[MemoryItem] = []

        if strategy == "importance_based":
            threshold = kwargs.get("importance_threshold", 0.3)
            for memory in self.memories.values():
                for item in memory.get_all_items():
                    if item.importance < threshold and memory.delete(item.id):
                        deleted_count += 1
                        deleted_items.append(item)

        elif strategy == "time_based":
            from datetime import timedelta

            days_old = kwargs.get("days_old", 30)
            cutoff_date = datetime.now() - timedelta(days=days_old)

            for memory in self.memories.values():
                for item in memory.get_all_items():
                    if item.created_at < cutoff_date and memory.delete(item.id):
                        deleted_count += 1
                        deleted_items.append(item)

        elif strategy == "capacity_based":
            max_items = kwargs.get("max_items", 1000)
            current_count = sum(len(memory.get_all_items()) for memory in self.memories.values())

            if current_count > max_items:
                all_items: List[MemoryItem] = []
                for memory in self.memories.values():
                    all_items.extend(memory.get_all_items())

                all_items.sort(key=lambda item: item.importance)
                items_to_delete = current_count - max_items

                for item in all_items[:items_to_delete]:
                    if self.memories[item.memory_type].delete(item.id):
                        deleted_count += 1
                        deleted_items.append(item)

        elif strategy == "combined":
            importance_result = self.forget("importance_based", **kwargs)
            time_result = self.forget("time_based", **kwargs)

            deleted_count = (
                importance_result.get("deleted_count", 0) + time_result.get("deleted_count", 0)
            )
            deleted_items = (
                importance_result.get("deleted_items", [])
                + time_result.get("deleted_items", [])
            )

        return {
            "deleted_count": deleted_count,
            "deleted_items": [
                item.id if isinstance(item, MemoryItem) else item for item in deleted_items
            ],
            "strategy": strategy,
        }

    def consolidate(self, importance_threshold: float = 0.7) -> Dict[str, Any]:
        """巩固记忆：将重要的工作记忆转移到情景记忆。
        
        参数:
            importance_threshold: 重要性阈值
            
        返回:
            包含巩固统计信息的字典
        """
        working_memory = self.memories[MemoryType.WORKING]
        episodic_memory = self.memories[MemoryType.EPISODIC]

        consolidated_count = 0
        consolidated_items: List[str] = []

        for item in working_memory.get_all_items():
            if item.importance >= importance_threshold:
                episodic_item = MemoryItem(
                    id=self._generate_id(),
                    content=item.content,
                    memory_type=MemoryType.EPISODIC,
                    embedding=item.embedding,
                    metadata=item.metadata,
                    importance=item.importance,
                    emotion=item.emotion,
                    tags=item.tags,
                    location=item.location,
                    source=item.source,
                    created_at=item.created_at,
                )

                episodic_memory.store(episodic_item)
                working_memory.delete(item.id)

                consolidated_count += 1
                consolidated_items.append(item.id)

        return {
            "consolidated_count": consolidated_count,
            "consolidated_items": consolidated_items,
            "importance_threshold": importance_threshold,
        }

    def get_summary(self, memory_type: Optional[MemoryType] = None) -> Dict[str, Any]:
        """获取记忆摘要信息。
        
        参数:
            memory_type: 记忆类型，默认为所有类型
            
        返回:
            包含摘要信息的字典
        """
        if memory_type:
            memory = self.memories[memory_type]
            items = memory.get_all_items()
            total_count = len(items)
            avg_importance = sum(item.importance for item in items) / total_count if total_count else 0
            recent_items = sorted(items, key=lambda item: item.created_at, reverse=True)[:5]

            return {
                "memory_type": memory_type.value,
                "total_count": total_count,
                "avg_importance": round(avg_importance, 3),
                "recent_items": [item.content[:50] + "..." for item in recent_items],
            }

        return {memory_type.value: self.get_summary(memory_type) for memory_type in MemoryType}

    def _link_episodic_to_semantic(self, episodic_item: MemoryItem):
        """将情景记忆链接到语义记忆（占位实现）。
        
        参数:
            episodic_item: 情景记忆项
        """
        del episodic_item
