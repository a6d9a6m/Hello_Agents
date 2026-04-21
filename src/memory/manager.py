"""记忆管理器（统一协调调度）"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime

from src.memory.base import BaseMemory, MemoryItem, MemoryType, MemoryConfig
from src.memory.embedding import TFIDFEmbedding
from src.memory.types.working import WorkingMemory
from src.memory.types.episodic import EpisodicMemory
from src.memory.types.semantic import SemanticMemory
from src.memory.types.perceptual import PerceptualMemory


class MemoryManager:
    """记忆管理器：统一协调各种记忆类型"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        # 使用TFIDF嵌入作为默认实现
        self.embedding_service = TFIDFEmbedding(self.config)
        
        # 初始化各种记忆类型
        self.memories: Dict[MemoryType, BaseMemory] = {
            MemoryType.WORKING: WorkingMemory(self.config),
            MemoryType.EPISODIC: EpisodicMemory(self.config, self.embedding_service),
            MemoryType.SEMANTIC: SemanticMemory(self.config, self.embedding_service),
            MemoryType.PERCEPTUAL: PerceptualMemory(self.config, self.embedding_service),
        }
    
    def store(self, content: str, memory_type: MemoryType, **metadata) -> str:
        """存储记忆"""
        # 生成嵌入向量
        embedding = self.embedding_service.embed(content)
        
        # 提取元数据中的上下文信息
        importance = metadata.pop("importance", 0.5)
        emotion = metadata.pop("emotion", None)
        tags = metadata.pop("tags", [])
        location = metadata.pop("location", None)
        source = metadata.pop("source", None)
        
        # 创建记忆项
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
            source=source
        )
        
        # 存储到对应的记忆系统
        memory_id = self.memories[memory_type].store(item)
        
        # 如果是情景记忆，可能需要关联到语义记忆
        if memory_type == MemoryType.EPISODIC:
            self._link_episodic_to_semantic(item)
        
        return memory_id
    
    def retrieve(self, query: str, memory_types: Optional[List[MemoryType]] = None, limit: int = 10) -> List[MemoryItem]:
        """检索记忆"""
        if memory_types is None:
            memory_types = list(MemoryType)
        
        results = []
        for memory_type in memory_types:
            memory = self.memories[memory_type]
            items = memory.retrieve(query, limit=limit)
            results.extend(items)
        
        # 按相关性排序
        query_embedding = self.embedding_service.embed(query)
        results.sort(
            key=lambda x: self.embedding_service.cosine_similarity(query_embedding, x.embedding or []),
            reverse=True
        )
        
        return results[:limit]
    
    def retrieve_context(self, query: str, context_size: int = 5) -> str:
        """检索上下文信息"""
        memories = self.retrieve(query, limit=context_size)
        
        # 构建上下文字符串
        context_parts = []
        for i, memory in enumerate(memories, 1):
            context_parts.append(f"[{memory.memory_type.value}] {memory.content}")
        
        return "\n".join(context_parts)
    
    def delete(self, memory_id: str, memory_type: MemoryType) -> bool:
        """删除记忆"""
        return self.memories[memory_type].delete(memory_id)
    
    def update(self, memory_id: str, memory_type: MemoryType, updates: Dict[str, Any]) -> bool:
        """更新记忆"""
        return self.memories[memory_type].update(memory_id, updates)
    
    def clear(self, memory_type: Optional[MemoryType] = None) -> bool:
        """清空记忆"""
        if memory_type:
            return self.memories[memory_type].clear()
        else:
            success = True
            for memory in self.memories.values():
                success = success and memory.clear()
            return success
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {}
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
                avg_importance = 0
            
            # 按情感标签统计
            emotion_stats = {}
            for item in items:
                if item.emotion:
                    emotion_stats[item.emotion] = emotion_stats.get(item.emotion, 0) + 1
            
            stats[memory_type.value] = {
                "count": count,
                "avg_importance": round(avg_importance, 3),
                "emotion_distribution": emotion_stats,
                "recent_count": len([item for item in items 
                                   if (datetime.now() - item.created_at).days < 7]),
                "last_updated": datetime.now().isoformat()
            }
        
        # 总体统计
        overall_avg_importance = total_importance / total_count if total_count > 0 else 0
        stats["overall"] = {
            "total_count": total_count,
            "avg_importance": round(overall_avg_importance, 3),
            "memory_types": len(MemoryType)
        }
        
        return stats
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        import uuid
        return str(uuid.uuid4())
    
    def search(self, query: str, memory_types: Optional[List[MemoryType]] = None, 
               search_mode: str = "hybrid", limit: int = 10, 
               min_importance: float = 0.0) -> List[MemoryItem]:
        """搜索记忆（支持多种搜索模式）"""
        if memory_types is None:
            memory_types = list(MemoryType)
        
        all_results = []
        
        for memory_type in memory_types:
            memory = self.memories[memory_type]
            
            # 根据搜索模式调用不同的检索方法
            if search_mode == "keyword":
                items = memory.search_by_keyword(query, limit=limit)
            elif search_mode == "semantic":
                items = memory.search_by_semantic(query, limit=limit)
            elif search_mode == "hybrid":
                items = memory.search_hybrid(query, limit=limit)
            else:
                items = memory.retrieve(query, limit=limit)
            
            # 过滤重要性
            items = [item for item in items if item.importance >= min_importance]
            all_results.extend(items)
        
        # 按相关性排序
        query_embedding = self.embedding_service.embed(query)
        all_results.sort(
            key=lambda x: self.embedding_service.cosine_similarity(query_embedding, x.embedding or []),
            reverse=True
        )
        
        return all_results[:limit]
    
    def forget(self, strategy: str = "importance_based", **kwargs) -> Dict[str, Any]:
        """遗忘记忆（支持多种策略）"""
        deleted_count = 0
        deleted_items = []
        
        if strategy == "importance_based":
            # 基于重要性的遗忘
            threshold = kwargs.get("importance_threshold", 0.3)
            for memory_type, memory in self.memories.items():
                items = memory.get_all_items()
                for item in items:
                    if item.importance < threshold:
                        if memory.delete(item.id):
                            deleted_count += 1
                            deleted_items.append(item)
        
        elif strategy == "time_based":
            # 基于时间的遗忘
            from datetime import datetime, timedelta
            days_old = kwargs.get("days_old", 30)
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            for memory_type, memory in self.memories.items():
                items = memory.get_all_items()
                for item in items:
                    if item.created_at < cutoff_date:
                        if memory.delete(item.id):
                            deleted_count += 1
                            deleted_items.append(item)
        
        elif strategy == "capacity_based":
            # 基于容量的遗忘
            max_items = kwargs.get("max_items", 1000)
            current_count = sum(len(memory.get_all_items()) for memory in self.memories.values())
            
            if current_count > max_items:
                # 收集所有记忆并按重要性排序
                all_items = []
                for memory_type, memory in self.memories.items():
                    all_items.extend(memory.get_all_items())
                
                all_items.sort(key=lambda x: x.importance)
                
                # 删除最不重要的记忆直到达到容量限制
                items_to_delete = current_count - max_items
                for i in range(items_to_delete):
                    if i < len(all_items):
                        item = all_items[i]
                        memory_type = item.memory_type
                        if self.memories[memory_type].delete(item.id):
                            deleted_count += 1
                            deleted_items.append(item)
        
        elif strategy == "combined":
            # 组合策略
            importance_result = self.forget("importance_based", **kwargs)
            time_result = self.forget("time_based", **kwargs)
            
            deleted_count = importance_result.get("deleted_count", 0) + time_result.get("deleted_count", 0)
            deleted_items = importance_result.get("deleted_items", []) + time_result.get("deleted_items", [])
        
        return {
            "deleted_count": deleted_count,
            "deleted_items": [item.id for item in deleted_items],
            "strategy": strategy
        }
    
    def consolidate(self, importance_threshold: float = 0.7) -> Dict[str, Any]:
        """整合记忆（工作记忆→情景记忆）"""
        working_memory = self.memories[MemoryType.WORKING]
        episodic_memory = self.memories[MemoryType.EPISODIC]
        
        working_items = working_memory.get_all_items()
        consolidated_count = 0
        consolidated_items = []
        
        for item in working_items:
            if item.importance >= importance_threshold:
                # 转换为情景记忆
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
                    created_at=item.created_at
                )
                
                # 存储到情景记忆
                episodic_memory.store(episodic_item)
                
                # 从工作记忆中删除
                working_memory.delete(item.id)
                
                consolidated_count += 1
                consolidated_items.append(item.id)
        
        return {
            "consolidated_count": consolidated_count,
            "consolidated_items": consolidated_items,
            "importance_threshold": importance_threshold
        }
    
    def get_summary(self, memory_type: Optional[MemoryType] = None) -> Dict[str, Any]:
        """获取记忆摘要"""
        if memory_type:
            memory = self.memories[memory_type]
            items = memory.get_all_items()
            
            # 计算摘要统计
            total_count = len(items)
            avg_importance = sum(item.importance for item in items) / total_count if total_count > 0 else 0
            recent_items = sorted(items, key=lambda x: x.created_at, reverse=True)[:5]
            
            return {
                "memory_type": memory_type.value,
                "total_count": total_count,
                "avg_importance": round(avg_importance, 3),
                "recent_items": [item.content[:50] + "..." for item in recent_items]
            }
        else:
            # 所有记忆类型的摘要
            summary = {}
            for mt in MemoryType:
                summary[mt.value] = self.get_summary(mt)
            return summary
    
    def _link_episodic_to_semantic(self, episodic_item: MemoryItem):
        """将情景记忆关联到语义记忆"""
        # 这里实现从情景记忆中提取实体和关系，存储到语义记忆
        # 待实现：实体识别和关系提取
        pass