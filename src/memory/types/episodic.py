"""情景记忆实现。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.memory.base import BaseMemory, MemoryConfig, MemoryItem, MemoryType
from src.memory.embedding import EmbeddingService
from src.memory.storage.document_store import DocumentStore
from src.memory.storage.qdrant_store import QdrantStore


class EpisodicMemory(BaseMemory):
    """情景记忆：由 manager 管理，但底层使用 SQLite + Qdrant 混合存储。

    设计目标：
        1. SQLite 始终作为主事实来源，负责结构化字段、时间线和关键词过滤。
        2. Qdrant 只负责向量检索增强；不可用时自动优雅降级，不影响基础读写。
        3. 对外暴露全量、关键词、语义、混合检索，便于 manager 统一调度。
    """

    _SEMANTIC_WEIGHT = 0.8
    _RECENCY_WEIGHT = 0.2

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        super().__init__(config)
        self.embedding_service = embedding_service
        self.vector_store = QdrantStore(config) if config else None
        self.document_store = DocumentStore(config) if config else None

    def store(self, item: MemoryItem) -> str:
        """存储情景记忆项。"""
        if item.embedding is None and self.embedding_service:
            item.embedding = self.embedding_service.embed(item.content)

        document_success = True
        if self.document_store:
            document_success = self.document_store.store_document(
                doc_id=item.id,
                content=item.content,
                doc_type="episodic",
                metadata=self._build_document_metadata(item),
                created_at=item.created_at,
                updated_at=item.updated_at,
            )

        # 向量存储失败不影响整体可用性，因此只作为增强路径处理。
        if self.vector_store and item.embedding:
            self.vector_store.store_vector(
                vector_id=item.id,
                vector=item.embedding,
                payload={
                    "content": item.content,
                    "type": item.memory_type.value,
                    "metadata": self._build_document_metadata(item),
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat(),
                },
            )

        if not document_success:
            raise RuntimeError("情景记忆写入 SQLite 失败")

        return item.id

    def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """默认检索入口，优先走混合检索，空查询时返回最近时间线。"""
        if not query.strip():
            items = self.get_all_items()
            items.sort(key=lambda item: (item.created_at, item.id))
            return items[:limit]
        return self.search_hybrid(query, limit=limit)

    def delete(self, memory_id: str) -> bool:
        """删除情景记忆项。"""
        success = True

        if self.document_store:
            success = self.document_store.delete(memory_id) and success

        # Qdrant 删除失败不应阻断主流程，避免降级场景下出现“删不掉”的假失败。
        if self.vector_store:
            self.vector_store.delete(memory_id)

        return success

    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """更新情景记忆项。"""
        if not self.document_store:
            return False

        current = self.document_store.get_document(memory_id)
        if current is None:
            return False

        metadata = current.get("metadata", {}).copy()
        content = updates.get("content", current.get("content", ""))

        for field in ("importance", "emotion", "tags", "location", "source", "ttl"):
            if field in updates:
                metadata[field] = updates[field]

        if "metadata" in updates and isinstance(updates["metadata"], dict):
            metadata["extra_metadata"] = updates["metadata"]

        if "event_time" in updates:
            event_time = updates["event_time"]
            metadata["event_time"] = (
                event_time.isoformat() if isinstance(event_time, datetime) else str(event_time)
            )

        if "content" in updates and self.embedding_service:
            metadata["embedding"] = self.embedding_service.embed(content)

        updated = self.document_store.update_document(
            memory_id,
            content=content,
            metadata=metadata,
        )
        if not updated:
            return False

        refreshed = self._document_to_memory_item(self.document_store.get_document(memory_id))
        if refreshed and self.vector_store and refreshed.embedding:
            self.vector_store.store_vector(
                vector_id=refreshed.id,
                vector=refreshed.embedding,
                payload={
                    "content": refreshed.content,
                    "type": refreshed.memory_type.value,
                    "metadata": self._build_document_metadata(refreshed),
                    "created_at": refreshed.created_at.isoformat(),
                    "updated_at": refreshed.updated_at.isoformat(),
                },
            )

        return True

    def clear(self) -> bool:
        """清空情景记忆。"""
        success = True

        if self.document_store:
            success = self.document_store.clear() and success

        if self.vector_store:
            self.vector_store.clear()

        return success

    def get_timeline(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[MemoryItem]:
        """按时间线返回事件。

        说明:
            - 时间线查询以 SQLite 为主，因为它更适合做时间范围过滤和稳定排序。
            - 结果按创建时间升序返回，尽量保持事件的先后关系。
        """
        if not self.document_store:
            return []

        documents = self.document_store.list_documents_by_time_range(
            start_time=start_time,
            end_time=end_time,
            doc_type="episodic",
        )
        return [item for item in (self._document_to_memory_item(doc) for doc in documents) if item]

    def get_all_items(self) -> List[MemoryItem]:
        """获取全部情景记忆项。"""
        if not self.document_store:
            return []

        documents = self.document_store.list_documents(doc_type="episodic", limit=10000)
        return [item for item in (self._document_to_memory_item(doc) for doc in documents) if item]

    def search_by_keyword(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """关键词搜索。"""
        if not self.document_store or limit <= 0:
            return []

        documents = self.document_store.search_documents_by_metadata_keyword(
            query=query,
            doc_type="episodic",
            limit=max(limit * 3, limit),
        )
        scored_items = self._score_documents_by_keyword(query, documents)
        return scored_items[:limit]

    def search_by_semantic(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """语义搜索。

        说明:
            - 若 Qdrant 和 embedding 服务均可用，则使用向量结果并套用统一评分公式。
            - 若任一环节不可用，则优雅降级为 SQLite 关键词搜索，保证基础功能不丢失。
        """
        if limit <= 0:
            return []

        vector_results = self._search_vector(query, limit=max(limit * 3, limit))
        if vector_results:
            return vector_results[:limit]

        return self.search_by_keyword(query, limit=limit)

    def search_hybrid(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """混合搜索。"""
        if limit <= 0:
            return []

        documents: Dict[str, Dict[str, Any]] = {}
        keyword_documents = []

        if self.document_store:
            keyword_documents = self.document_store.search_documents_by_metadata_keyword(
                query=query,
                doc_type="episodic",
                limit=max(limit * 4, limit),
            )
            for document in keyword_documents:
                documents[document["id"]] = document

        vector_items = self._search_vector(query, limit=max(limit * 4, limit))
        for item in vector_items:
            if item.id not in documents and self.document_store:
                document = self.document_store.get_document(item.id)
                if document:
                    documents[item.id] = document

        scored_items = []
        for document in documents.values():
            item = self._document_to_memory_item(document)
            if item is None:
                continue

            semantic_similarity = self._semantic_similarity_from_item(query, item)
            keyword_similarity = self._keyword_similarity(query, item)
            similarity = max(semantic_similarity, keyword_similarity)
            if similarity <= 0:
                continue

            score = self._final_score(item, semantic_similarity)
            scored_items.append((score, item.created_at.timestamp(), item))

        scored_items.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item for _, _, item in scored_items[:limit]]

    def score_item(self, query: str, item: MemoryItem) -> Optional[float]:
        """为 manager 提供情景记忆内部评分。"""
        if item.memory_type != MemoryType.EPISODIC:
            return None

        semantic_similarity = self._semantic_similarity_from_item(query, item)
        keyword_similarity = self._keyword_similarity(query, item)
        similarity = max(semantic_similarity, keyword_similarity)
        if similarity <= 0:
            return 0.0

        return self._final_score(item, semantic_similarity)

    def link_events(self, event1_id: str, event2_id: str, relation: str = "related"):
        """链接两个事件。"""
        if self.document_store:
            self.document_store.create_relation(
                from_id=event1_id,
                to_id=event2_id,
                relation_type=relation,
            )

    def _build_document_metadata(self, item: MemoryItem) -> Dict[str, Any]:
        """构建落库元数据。

        说明:
            - 将结构化字段一并写入 SQLite，便于后续复杂过滤和时间序列查询。
            - `event_time` 优先从 metadata 中提取；若没有，则回退到 created_at，
              这样时间近因性既能反映事件真实发生时间，也能兼容未显式提供事件时间的旧数据。
        """
        event_time = item.metadata.get("event_time") if isinstance(item.metadata, dict) else None
        if isinstance(event_time, datetime):
            event_time_text = event_time.isoformat()
        elif isinstance(event_time, str) and event_time:
            event_time_text = event_time
        else:
            event_time_text = item.created_at.isoformat()

        return {
            "memory_type": item.memory_type.value,
            "importance": item.importance,
            "emotion": item.emotion,
            "tags": item.tags,
            "location": item.location,
            "source": item.source,
            "ttl": item.ttl,
            "event_time": event_time_text,
            "embedding": item.embedding,
            "extra_metadata": item.metadata,
        }

    def _document_to_memory_item(self, document: Optional[Dict[str, Any]]) -> Optional[MemoryItem]:
        """将 SQLite 文档记录恢复为 MemoryItem。"""
        if not document:
            return None

        metadata = document.get("metadata", {}) or {}
        extra_metadata = metadata.get("extra_metadata", {}) or {}
        item_metadata = dict(extra_metadata)
        if metadata.get("event_time"):
            item_metadata["event_time"] = metadata.get("event_time")

        return MemoryItem(
            id=document.get("id", ""),
            content=document.get("content", ""),
            memory_type=MemoryType.EPISODIC,
            embedding=metadata.get("embedding"),
            metadata=item_metadata,
            importance=float(metadata.get("importance", 0.5) or 0.5),
            emotion=metadata.get("emotion"),
            tags=list(metadata.get("tags", []) or []),
            location=metadata.get("location"),
            source=metadata.get("source"),
            ttl=metadata.get("ttl"),
            created_at=self._parse_datetime(document.get("created_at")),
            updated_at=self._parse_datetime(document.get("updated_at")),
        )

    def _search_vector(self, query: str, limit: int) -> List[MemoryItem]:
        """执行向量检索并根据统一公式重排。"""
        if not self.vector_store or not self.embedding_service or not query.strip():
            return []

        if not self.vector_store.is_available():
            return []

        query_embedding = self.embedding_service.embed(query)
        vector_results = self.vector_store.search(query_vector=query_embedding, limit=limit)
        scored_items = []

        for result in vector_results:
            item = self._vector_result_to_memory_item(result)
            if item is None:
                continue

            semantic_similarity = self._normalize_similarity_score(result.get("score", 0.0))
            score = self._final_score(item, semantic_similarity)
            scored_items.append((score, item.created_at.timestamp(), item))

        scored_items.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item for _, _, item in scored_items]

    def _vector_result_to_memory_item(self, result: Dict[str, Any]) -> Optional[MemoryItem]:
        """将 Qdrant 结果转换为 MemoryItem。

        说明:
            - 优先回到 SQLite 取全量结构化数据，保证情景记忆字段完整、时间顺序稳定。
            - 若 SQLite 中暂时没有该记录，再退回使用 Qdrant payload 的最小字段集合。
        """
        memory_id = str(result.get("id", ""))
        if self.document_store and memory_id:
            document = self.document_store.get_document(memory_id)
            if document:
                return self._document_to_memory_item(document)

        payload = result.get("payload", {}) or {}
        metadata = payload.get("metadata", {}) or {}
        extra_metadata = metadata.get("extra_metadata", {}) or {}
        item_metadata = dict(extra_metadata)
        if metadata.get("event_time"):
            item_metadata["event_time"] = metadata.get("event_time")
        return MemoryItem(
            id=memory_id,
            content=payload.get("content", ""),
            memory_type=MemoryType.EPISODIC,
            embedding=result.get("vector") or metadata.get("embedding"),
            metadata=item_metadata,
            importance=float(metadata.get("importance", 0.5) or 0.5),
            emotion=metadata.get("emotion"),
            tags=list(metadata.get("tags", []) or []),
            location=metadata.get("location"),
            source=metadata.get("source"),
            ttl=metadata.get("ttl"),
            created_at=self._parse_datetime(payload.get("created_at")),
            updated_at=self._parse_datetime(payload.get("updated_at")),
        )

    def _score_documents_by_keyword(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[MemoryItem]:
        """对 SQLite 关键词结果按统一公式评分。"""
        scored_items = []
        for document in documents:
            item = self._document_to_memory_item(document)
            if item is None:
                continue

            keyword_similarity = self._keyword_similarity(query, item)
            if keyword_similarity <= 0:
                continue

            score = self._final_score(item, 0.0, keyword_similarity=keyword_similarity)
            scored_items.append((score, item.created_at.timestamp(), item))

        scored_items.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item for _, _, item in scored_items]

    def _final_score(
        self,
        item: MemoryItem,
        semantic_similarity: float,
        *,
        keyword_similarity: float = 0.0,
    ) -> float:
        """按要求实现情景记忆统一评分公式。

        公式严格为:
            `(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4)`

        说明:
            - `semantic_similarity` 表示向量相似度。
            - 当 Qdrant 不可用时，`semantic_similarity` 为 0，此时排序退化为“关键词召回 + 时间/重要性重排”。
            - `keyword_similarity` 不直接进入公式，只用来决定一条记录是否应被保留在结果集中。
        """
        if semantic_similarity <= 0 and keyword_similarity <= 0:
            return 0.0

        importance = min(max(item.importance, 0.0), 1.0)
        recency = self._time_recency(item)
        return (
            semantic_similarity * self._SEMANTIC_WEIGHT
            + recency * self._RECENCY_WEIGHT
        ) * (0.8 + importance * 0.4)

    def _time_recency(self, item: MemoryItem) -> float:
        """基于事件时间或创建时间计算时间近因性。

        说明:
            - 若 metadata 中显式携带 `event_time`，优先使用它，因为它更接近事件真实发生时刻。
            - 若没有事件时间，则回退到 `created_at`，兼容旧数据与普通写入流程。
            - 采用 `1 / (1 + age_days)` 的平滑衰减：当天事件权重接近 1，
              随着天数增长缓慢下降，既能体现“最近发生更重要”，也不会让旧事件瞬间失效。
        """
        event_time = self._resolve_event_time(item)
        age_seconds = max((datetime.now() - event_time).total_seconds(), 0.0)
        age_days = age_seconds / 86400.0
        return 1.0 / (1.0 + age_days)

    def _resolve_event_time(self, item: MemoryItem) -> datetime:
        """解析事件时间，失败时回退到创建时间。"""
        if isinstance(item.metadata, dict):
            raw_event_time = item.metadata.get("event_time")
            if isinstance(raw_event_time, datetime):
                return raw_event_time
            if isinstance(raw_event_time, str) and raw_event_time:
                try:
                    return datetime.fromisoformat(raw_event_time)
                except ValueError:
                    pass
        return item.created_at

    def _semantic_similarity_from_item(self, query: str, item: MemoryItem) -> float:
        """从 MemoryItem 计算向量语义相似度。"""
        if not self.embedding_service or not item.embedding or not query.strip():
            return 0.0

        query_embedding = self.embedding_service.embed(query)
        return self.embedding_service.cosine_similarity(query_embedding, item.embedding)

    def _keyword_similarity(self, query: str, item: MemoryItem) -> float:
        """轻量关键词相似度。"""
        query_text = query.strip().lower()
        if not query_text:
            return 0.0

        haystack = self._item_search_text(item).lower()
        if query_text in haystack:
            return 1.0

        query_tokens = {token for token in query_text.split() if token}
        item_tokens = {token for token in haystack.split() if token}
        if not query_tokens or not item_tokens:
            return 0.0

        overlap = len(query_tokens.intersection(item_tokens))
        return overlap / len(query_tokens)

    def _item_search_text(self, item: MemoryItem) -> str:
        """拼接可检索文本，尽量保留事件完整上下文。"""
        parts = [item.content]
        if item.tags:
            parts.extend(item.tags)
        if item.location:
            parts.append(item.location)
        if item.source:
            parts.append(item.source)
        if isinstance(item.metadata, dict):
            parts.extend(str(value) for value in item.metadata.values() if value is not None)
        return " ".join(part for part in parts if part)

    def _normalize_similarity_score(self, raw_score: float) -> float:
        """将 Qdrant 返回分数夹紧到 0 到 1。"""
        return min(max(float(raw_score or 0.0), 0.0), 1.0)

    def _parse_datetime(self, value: Optional[str]) -> datetime:
        """解析时间字符串，失败时回退为当前时间。"""
        if not value:
            return datetime.now()
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now()
