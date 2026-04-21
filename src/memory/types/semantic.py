"""语义记忆实现"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.memory.base import BaseMemory, MemoryItem, MemoryType, MemoryConfig
from src.memory.embedding import EmbeddingService
from src.memory.storage.qdrant_store import QdrantStore
from src.memory.storage.neo4j_store import Neo4jStore


class SemanticMemory(BaseMemory):
    """语义记忆：知识图谱记忆"""
    
    def __init__(self, config: Optional[MemoryConfig] = None, embedding_service: Optional[EmbeddingService] = None):
        super().__init__(config)
        self.embedding_service = embedding_service
        self.vector_store = QdrantStore(config) if config else None
        self.graph_store = Neo4jStore(config) if config else None
    
    def store(self, item: MemoryItem) -> str:
        """存储语义记忆项"""
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
        
        # 提取实体和关系，存储到图数据库
        if self.graph_store:
            entities, relations = self._extract_knowledge(item.content)
            
            for entity in entities:
                self.graph_store.create_node(
                    node_id=f"{item.id}_{entity['name']}",
                    node_type=entity.get("type", "entity"),
                    properties={
                        "name": entity["name"],
                        "source_memory": item.id,
                        **entity.get("properties", {})
                    }
                )
            
            for relation in relations:
                self.graph_store.create_relation(
                    from_id=f"{item.id}_{relation['from']}",
                    to_id=f"{item.id}_{relation['to']}",
                    relation_type=relation["type"],
                    properties=relation.get("properties", {})
                )
        
        return item.id
    
    def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """检索语义记忆项"""
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
                    memory_type=MemoryType.SEMANTIC,
                    embedding=result.get("vector", []),
                    metadata=payload.get("metadata", {}),
                    created_at=datetime.fromisoformat(payload.get("created_at", datetime.now().isoformat())),
                    updated_at=datetime.fromisoformat(payload.get("updated_at", datetime.now().isoformat()))
                )
                results.append(item)
        
        return results
    
    def delete(self, memory_id: str) -> bool:
        """删除语义记忆项"""
        success = True
        
        if self.vector_store:
            success = success and self.vector_store.delete(memory_id)
        
        if self.graph_store:
            # 删除相关的图节点
            success = success and self.graph_store.delete_nodes_by_source(memory_id)
        
        return success
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """更新语义记忆项"""
        # 语义记忆更新较复杂，需要更新向量和图
        return False
    
    def clear(self) -> bool:
        """清空语义记忆"""
        success = True
        
        if self.vector_store:
            success = success and self.vector_store.clear()
        
        if self.graph_store:
            success = success and self.graph_store.clear()
        
        return success
    
    def query_knowledge_graph(self, query: str) -> List[Dict[str, Any]]:
        """查询知识图谱"""
        if not self.graph_store:
            return []
        
        # 这里可以支持Cypher查询或自然语言转Cypher
        return self.graph_store.query(query)
    
    def get_related_concepts(self, concept: str, depth: int = 2) -> List[Dict[str, Any]]:
        """获取相关概念"""
        if not self.graph_store:
            return []
        
        return self.graph_store.get_related_nodes(concept, depth)
    
    def _extract_knowledge(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """从文本中提取知识（实体和关系）"""
        # 这里应该使用NLP技术提取实体和关系
        # 简化实现：返回空列表
        entities = []
        relations = []
        
        # 示例：简单提取名词作为实体
        # 实际实现可以使用spaCy、NLTK或预训练模型
        
        return entities, relations