"""语义记忆实现。"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.memory.base import BaseMemory, MemoryItem, MemoryType, MemoryConfig
from src.memory.embedding import EmbeddingService
from src.memory.storage.qdrant_store import QdrantStore
from src.memory.storage.neo4j_store import Neo4jStore


class SemanticMemory(BaseMemory):
    """语义记忆：由 manager 管理，但底层采用 Qdrant + Neo4j 混合架构。

    设计约束：
        1. Qdrant 负责向量语义召回，是主检索路径。
        2. Neo4j 负责实体、关系和图相似度，是补充评分路径。
        3. 最终排序严格使用 `(向量相似度 × 0.7 + 图相似度 × 0.3) × (0.8 + 重要性 × 0.4)`。
        4. 重要性按 [0, 1] 夹紧后映射到 [0.8, 1.2]，保证权重范围稳定。
        5. Neo4j 不可用时，图相似度退化为 0，但基本记忆能力仍由 Qdrant 或本地缓存维持。
    """

    _VECTOR_WEIGHT = 0.7
    _GRAPH_WEIGHT = 0.3

    def __init__(self, config: Optional[MemoryConfig] = None, embedding_service: Optional[EmbeddingService] = None):
        super().__init__(config)
        self.embedding_service = embedding_service
        self.vector_store = QdrantStore(config) if config else None
        self.graph_store = Neo4jStore(config) if config else None
        self._items: Dict[str, MemoryItem] = {}
    
    def store(self, item: MemoryItem) -> str:
        """存储语义记忆项。"""
        # 先补齐 embedding，并把原始记忆放入本地索引，
        # 这样即使外部服务暂时不可用，也不会丢掉最基础的读写能力。
        if item.embedding is None and self.embedding_service:
            item.embedding = self.embedding_service.embed(item.content)
        self._items[item.id] = item

        entities, relations = self._extract_knowledge(item.content)
        payload = {
            "content": item.content,
            "type": item.memory_type.value,
            "metadata": {
                **(item.metadata or {}),
                "importance": item.importance,
                "emotion": item.emotion,
                "tags": item.tags,
                "location": item.location,
                "source": item.source,
                "ttl": item.ttl,
                "entities": entities,
                "relations": relations,
                "embedding": item.embedding,
            },
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat()
        }
        
        # 向量路径是语义记忆的主召回入口。
        if self.vector_store:
            self.vector_store.store_vector(
                vector_id=item.id,
                vector=item.embedding or [],
                payload=payload,
            )
        
        # 图路径只负责结构化知识表示与补充推理，不应影响基础可用性。
        if self.graph_store:
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
        """默认检索入口：空查询返回全部，有查询走混合检索。"""
        if limit <= 0:
            return []
        if not query.strip():
            return self.get_all_items()[:limit]
        return self.search_hybrid(query, limit=limit)
    
    def delete(self, memory_id: str) -> bool:
        """删除语义记忆项。"""
        success = True
        self._items.pop(memory_id, None)
        
        if self.vector_store:
            success = success and self.vector_store.delete(memory_id)
        
        if self.graph_store:
            # 删除相关的图节点
            success = success and self.graph_store.delete_nodes_by_source(memory_id)
        
        return success
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """更新语义记忆项。"""
        current = self._items.get(memory_id)
        if current is None:
            return False

        updated_item = MemoryItem(
            id=current.id,
            content=updates.get("content", current.content),
            memory_type=current.memory_type,
            embedding=current.embedding,
            metadata=updates.get("metadata", current.metadata),
            created_at=current.created_at,
            updated_at=datetime.now(),
            ttl=updates.get("ttl", current.ttl),
            importance=float(updates.get("importance", current.importance)),
            emotion=updates.get("emotion", current.emotion),
            tags=list(updates.get("tags", current.tags)),
            location=updates.get("location", current.location),
            source=updates.get("source", current.source),
        )

        self.delete(memory_id)
        self.store(updated_item)
        return True
    
    def clear(self) -> bool:
        """清空语义记忆。"""
        success = True
        self._items.clear()
        
        if self.vector_store:
            success = success and self.vector_store.clear()
        
        if self.graph_store:
            success = success and self.graph_store.clear()
        
        return success
    
    def query_knowledge_graph(self, query: str) -> List[Dict[str, Any]]:
        """查询知识图谱。"""
        if not self.graph_store:
            return []
        
        # 这里可以支持Cypher查询或自然语言转Cypher
        return self.graph_store.query(query)
    
    def get_related_concepts(self, concept: str, depth: int = 2) -> List[Dict[str, Any]]:
        """获取相关概念。"""
        if not self.graph_store:
            return []
        
        return self.graph_store.get_related_nodes(concept, depth)
    
    def get_all_items(self) -> List[MemoryItem]:
        """获取所有语义记忆项。"""
        items = list(self._items.values())
        items.sort(key=lambda item: (item.created_at, item.id))
        return items
    
    def search_by_keyword(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """关键词搜索。"""
        if limit <= 0:
            return []

        query_text = query.strip().lower()
        if not query_text:
            return self.get_all_items()[:limit]

        graph_scores = self._graph_scores(query, limit=max(limit * 4, limit))
        scored_items = []
        for item in self._items.values():
            keyword_similarity = self._keyword_similarity(query, item)
            if keyword_similarity <= 0 and graph_scores.get(item.id, 0.0) <= 0:
                continue

            score = self._hybrid_score(
                vector_similarity=0.0,
                graph_similarity=max(keyword_similarity, graph_scores.get(item.id, 0.0)),
                importance=item.importance,
            )
            scored_items.append((score, item.updated_at.timestamp(), item))

        scored_items.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item for _, _, item in scored_items[:limit]]
    
    def search_by_semantic(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """语义搜索。"""
        if limit <= 0:
            return []

        vector_results = self._search_vector(query, limit=max(limit * 4, limit))
        if vector_results:
            return vector_results[:limit]

        # 当向量服务缺失时，退化到关键词搜索，保证基础记忆能力存在。
        return self.search_by_keyword(query, limit=limit)

    def search_hybrid(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """混合搜索。"""
        if limit <= 0:
            return []

        candidate_map: Dict[str, MemoryItem] = {}

        for item in self._search_vector(query, limit=max(limit * 4, limit)):
            candidate_map[item.id] = item

        for item in self.search_by_keyword(query, limit=max(limit * 4, limit)):
            candidate_map[item.id] = item

        graph_scores = self._graph_scores(query, limit=max(limit * 4, limit))
        scored_items = []
        for item in candidate_map.values():
            vector_similarity = self._semantic_similarity_from_item(query, item)
            graph_similarity = max(graph_scores.get(item.id, 0.0), self._keyword_similarity(query, item))
            if vector_similarity <= 0 and graph_similarity <= 0:
                continue

            score = self._hybrid_score(
                vector_similarity=vector_similarity,
                graph_similarity=graph_similarity,
                importance=item.importance,
            )
            scored_items.append((score, item.updated_at.timestamp(), item))

        scored_items.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item for _, _, item in scored_items[:limit]]

    def score_item(self, query: str, item: MemoryItem) -> Optional[float]:
        """为 manager 提供语义记忆内部评分。"""
        if item.memory_type != MemoryType.SEMANTIC:
            return None

        vector_similarity = self._semantic_similarity_from_item(query, item)
        graph_similarity = max(self._graph_scores(query, limit=100).get(item.id, 0.0), self._keyword_similarity(query, item))
        if vector_similarity <= 0 and graph_similarity <= 0:
            return 0.0

        return self._hybrid_score(
            vector_similarity=vector_similarity,
            graph_similarity=graph_similarity,
            importance=item.importance,
        )
    
    def _extract_knowledge(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """从文本中提取知识（实体和关系）。

        说明:
            - 这里采用轻量启发式而不是完整 NLP 管线，目的是先把结构补齐。
            - 实体抽取优先保留连续英文词、数字标识和常见中文专有片段。
            - 关系抽取识别最常见的“是/属于/包含/使用/依赖/连接/导致”等模式。
        """
        entities = []
        relations = []

        seen_entities = set()
        for match in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}", text):
            entity_name = match.strip("，。；：、（）()[]{} \t\n")
            if len(entity_name) < 2:
                continue
            normalized = entity_name.lower()
            if normalized in seen_entities:
                continue
            seen_entities.add(normalized)
            entities.append({"name": entity_name, "type": self._infer_entity_type(entity_name)})

        relation_patterns = [
            (r"([\u4e00-\u9fffA-Za-z0-9_\-]+)是([\u4e00-\u9fffA-Za-z0-9_\-]+)", "is_a"),
            (r"([\u4e00-\u9fffA-Za-z0-9_\-]+)属于([\u4e00-\u9fffA-Za-z0-9_\-]+)", "belongs_to"),
            (r"([\u4e00-\u9fffA-Za-z0-9_\-]+)包含([\u4e00-\u9fffA-Za-z0-9_\-]+)", "contains"),
            (r"([\u4e00-\u9fffA-Za-z0-9_\-]+)使用([\u4e00-\u9fffA-Za-z0-9_\-]+)", "uses"),
            (r"([\u4e00-\u9fffA-Za-z0-9_\-]+)依赖([\u4e00-\u9fffA-Za-z0-9_\-]+)", "depends_on"),
            (r"([\u4e00-\u9fffA-Za-z0-9_\-]+)连接([\u4e00-\u9fffA-Za-z0-9_\-]+)", "connects_to"),
            (r"([\u4e00-\u9fffA-Za-z0-9_\-]+)导致([\u4e00-\u9fffA-Za-z0-9_\-]+)", "causes"),
        ]

        seen_relations = set()
        for pattern, relation_type in relation_patterns:
            for from_name, to_name in re.findall(pattern, text):
                relation_key = (from_name.lower(), to_name.lower(), relation_type)
                if relation_key in seen_relations:
                    continue
                seen_relations.add(relation_key)
                relations.append({"from": from_name, "to": to_name, "type": relation_type})

        return entities, relations

    def _search_vector(self, query: str, limit: int) -> List[MemoryItem]:
        """执行向量检索，并按混合公式重排。"""
        if not self.vector_store or not self.embedding_service or not query.strip() or limit <= 0:
            return []

        query_embedding = self.embedding_service.embed(query)
        vector_results = self.vector_store.search(query_vector=query_embedding, limit=limit)
        graph_scores = self._graph_scores(query, limit=limit)
        scored_items = []

        for result in vector_results:
            item = self._vector_result_to_memory_item(result)
            if item is None:
                continue

            vector_similarity = self._normalize_similarity_score(result.get("score", 0.0))
            graph_similarity = graph_scores.get(item.id, 0.0)
            score = self._hybrid_score(
                vector_similarity=vector_similarity,
                graph_similarity=graph_similarity,
                importance=item.importance,
            )
            scored_items.append((score, item.updated_at.timestamp(), item))

        scored_items.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item for _, _, item in scored_items]

    def _vector_result_to_memory_item(self, result: Dict[str, Any]) -> Optional[MemoryItem]:
        """将向量检索结果恢复为 MemoryItem。"""
        memory_id = str(result.get("id", ""))
        if memory_id in self._items:
            return self._items[memory_id]

        payload = result.get("payload", {}) or {}
        metadata = payload.get("metadata", {}) or {}
        item = MemoryItem(
            id=memory_id,
            content=payload.get("content", ""),
            memory_type=MemoryType.SEMANTIC,
            embedding=result.get("vector") or metadata.get("embedding"),
            metadata={
                key: value
                for key, value in metadata.items()
                if key not in {"importance", "emotion", "tags", "location", "source", "ttl", "embedding", "entities", "relations"}
            },
            importance=float(metadata.get("importance", 0.5) or 0.5),
            emotion=metadata.get("emotion"),
            tags=list(metadata.get("tags", []) or []),
            location=metadata.get("location"),
            source=metadata.get("source"),
            ttl=metadata.get("ttl"),
            created_at=self._parse_datetime(payload.get("created_at")),
            updated_at=self._parse_datetime(payload.get("updated_at")),
        )
        self._items[item.id] = item
        return item

    def _graph_scores(self, query: str, limit: int) -> Dict[str, float]:
        """计算图相似度分数。

        说明:
            - 向量检索仍然是主路径，因此图分只作为补充信号。
            - 当 Neo4j 服务不可用时，本地轻量图索引仍可工作；
              若连图索引都无法提供命中，则自然退化为 0 分，不影响基础功能。
        """
        if not self.graph_store or limit <= 0:
            return {}

        query_entities, _ = self._extract_knowledge(query)
        entity_names = [entity.get("name", "") for entity in query_entities]
        return self.graph_store.score_memories_by_entities(entity_names, limit=limit)

    def _hybrid_score(self, vector_similarity: float, graph_similarity: float, importance: float) -> float:
        """按要求实现混合排序公式。"""
        normalized_importance = min(max(float(importance or 0.0), 0.0), 1.0)
        return (
            self._normalize_similarity_score(vector_similarity) * self._VECTOR_WEIGHT
            + self._normalize_similarity_score(graph_similarity) * self._GRAPH_WEIGHT
        ) * (0.8 + normalized_importance * 0.4)

    def _semantic_similarity_from_item(self, query: str, item: MemoryItem) -> float:
        """从 MemoryItem 计算向量相似度。"""
        if not self.embedding_service or not item.embedding or not query.strip():
            return 0.0

        query_embedding = self.embedding_service.embed(query)
        return self._normalize_similarity_score(
            self.embedding_service.cosine_similarity(query_embedding, item.embedding)
        )

    def _keyword_similarity(self, query: str, item: MemoryItem) -> float:
        """计算轻量关键词重合度。"""
        query_text = query.strip().lower()
        if not query_text:
            return 0.0

        haystack = self._item_search_text(item).lower()
        if query_text in haystack:
            return 1.0

        query_tokens = {token for token in re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]{1,}", query_text) if token}
        item_tokens = {token for token in re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]{1,}", haystack) if token}
        if not query_tokens or not item_tokens:
            return 0.0

        return len(query_tokens.intersection(item_tokens)) / len(query_tokens)

    def _item_search_text(self, item: MemoryItem) -> str:
        """拼接可检索文本。"""
        parts = [item.content]
        parts.extend(item.tags)
        if item.location:
            parts.append(item.location)
        if item.source:
            parts.append(item.source)
        if isinstance(item.metadata, dict):
            parts.extend(str(value) for value in item.metadata.values() if value is not None)
        return " ".join(part for part in parts if part)

    def _infer_entity_type(self, entity_name: str) -> str:
        """用轻量规则推断实体类型。"""
        lowered = entity_name.lower()
        if any(char.isdigit() for char in entity_name):
            return "identifier"
        if lowered.endswith(("系统", "平台", "数据库", "服务", "模型", "agent")):
            return "concept"
        if lowered.endswith(("公司", "团队", "部门", "组织")):
            return "organization"
        return "entity"

    def _normalize_similarity_score(self, raw_score: float) -> float:
        """将相似度夹紧到 0 到 1。"""
        return min(max(float(raw_score or 0.0), 0.0), 1.0)

    def _parse_datetime(self, value: Optional[str]) -> datetime:
        """解析时间字符串，失败时回退当前时间。"""
        if not value:
            return datetime.now()
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now()
