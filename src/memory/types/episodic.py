"""情景记忆实现"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime

from src.memory.base import BaseMemory, MemoryItem, MemoryType, MemoryConfig
from src.memory.embedding import EmbeddingService
from src.memory.storage.qdrant_store import QdrantStore
from src.memory.storage.document_store import DocumentStore


class EpisodicMemory(BaseMemory):
    """情景记忆：事件序列记忆"""
    
    def __init__(self, config: Optional[MemoryConfig] = None, embedding_service: Optional[EmbeddingService] = None):
        super().__init__(config)
        self.embedding_service = embedding_service
        self.vector_store = QdrantStore(config) if config else None
        self.document_store = DocumentStore(config) if config else None
    
    def store(self, item: MemoryItem) -> str:
        """存储情景记忆项"""
        # 确保有嵌入向量
        if item.embedding is None and self.embedding_service:
            item.embedding = self.embedding_service.embed(item.content)
        
        # 存储到向量数据库
        if self.vector_store:
            self.vector_store.store_vector(
                vector_id=item.id,
                vector=item.embedding or [],
                payload={
                    "content": item.content,
                    "type": item.memory_type.value,
                    "metadata": item.metadata,
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat()
                }
            )
        
        # 存储到文档数据库
        if self.document_store:
            self.document_store.store_document(
                doc_id=item.id,
                content=item.content,
                doc_type="episodic",
                metadata={
                    "memory_type": item.memory_type.value,
                    **item.metadata
                }
            )
        
        return item.id
    
    def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """检索情景记忆项"""
        results = []
        
        # 使用向量检索
        if self.vector_store and self.embedding_service:
            query_embedding = self.embedding_service.embed(query)
            vector_results = self.vector_store.search(
                query_vector=query_embedding,
                limit=limit
            )
            
            for result in vector_results:
                payload = result.get("payload", {})
                item = MemoryItem(
                    id=result.get("id", ""),
                    content=payload.get("content", ""),
                    memory_type=MemoryType.EPISODIC,
                    embedding=result.get("vector", []),
                    metadata=payload.get("metadata", {}),
                    created_at=datetime.fromisoformat(payload.get("created_at", datetime.now().isoformat())),
                    updated_at=datetime.fromisoformat(payload.get("updated_at", datetime.now().isoformat()))
                )
                results.append(item)
        
        return results
    
    def delete(self, memory_id: str) -> bool:
        """删除情景记忆项"""
        success = True
        
        if self.vector_store:
            success = success and self.vector_store.delete(memory_id)
        
        if self.document_store:
            success = success and self.document_store.delete(memory_id)
        
        return success
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """更新情景记忆项"""
        # 需要从存储中获取原始项
        # 这里简化实现，实际需要查询后更新
        return False
    
    def clear(self) -> bool:
        """清空情景记忆"""
        success = True
        
        if self.vector_store:
            success = success and self.vector_store.clear()
        
        if self.document_store:
            success = success and self.document_store.clear()
        
        return success
    
    def get_timeline(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[MemoryItem]:
        """获取时间线事件"""
        if not self.document_store:
            return []
        
        # 这里需要文档存储支持时间范围查询
        # 简化实现：返回所有项
        return self.retrieve("", limit=100)
    
    def link_events(self, event1_id: str, event2_id: str, relation: str = "related"):
        """链接两个事件"""
        if self.document_store:
            self.document_store.create_relation(
                from_id=event1_id,
                to_id=event2_id,
                relation_type=relation
            )