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
        
        # 创建记忆项
        item = MemoryItem(
            id=self._generate_id(),
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            metadata=metadata
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
        for memory_type, memory in self.memories.items():
            # 这里需要具体的记忆实现提供统计方法
            stats[memory_type.value] = {
                "count": 0,  # 待实现
                "last_updated": datetime.now().isoformat()
            }
        return stats
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _link_episodic_to_semantic(self, episodic_item: MemoryItem):
        """将情景记忆关联到语义记忆"""
        # 这里实现从情景记忆中提取实体和关系，存储到语义记忆
        # 待实现：实体识别和关系提取
        pass