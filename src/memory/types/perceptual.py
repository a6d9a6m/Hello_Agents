"""感知记忆实现。"""

from __future__ import annotations

import base64
import math
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from src.memory.base import BaseMemory, MemoryConfig, MemoryItem, MemoryType
from src.memory.embedding import EmbeddingService
from src.memory.storage.document_store import DocumentStore
from src.memory.storage.qdrant_store import QdrantStore


class PerceptualMemory(BaseMemory):
    """感知记忆：支持文本、图像、音频等模态的轻量实现。

    设计约束：
        1. manager 统一管理感知记忆，但不同模态使用独立向量集合，避免维度或语义混杂。
        2. 当前采用“文本桥接”的最小实现：图像/音频先依赖描述、转写或元数据形成桥接文本。
        3. 同模态检索只搜索目标模态集合；跨模态检索会并行搜索多个模态集合后统一排序。
        4. 当外部多模态模型不可用时，不阻塞主流程，至少保留文本存储和基于描述的跨模态检索。
    """

    _SIMILARITY_WEIGHT = 0.8
    _RECENCY_WEIGHT = 0.2
    _DEFAULT_HALF_LIFE_HOURS = 24.0 * 7.0
    _SUPPORTED_MODALITIES = ("text", "image", "audio")

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        super().__init__(config)
        self.embedding_service = embedding_service
        self.document_store = DocumentStore(self.config)
        self._vector_stores: Dict[str, QdrantStore] = {}
        self._items: Dict[str, MemoryItem] = {}
        self._item_modalities: Dict[str, str] = {}

    def store(
        self,
        item: MemoryItem,
        media_data: Optional[Union[bytes, str]] = None,
        media_type: str = "text",
    ) -> str:
        """存储感知记忆项。"""
        modality = self._normalize_media_type(media_type)
        bridge_text = self._build_bridge_text(
            content=item.content,
            media_type=modality,
            metadata=item.metadata,
            media_data=media_data,
        )

        # 感知记忆排序依赖向量相似度项；这里统一把文本内容或多模态描述文本嵌入。
        # 即使真实图像/音频模型缺失，也能依靠描述文本完成最小正确的跨模态检索。
        item.embedding = self._embed_text(bridge_text)
        item.metadata = {
            **(item.metadata or {}),
            "media_type": modality,
            "bridge_text": bridge_text,
            "has_media": media_data is not None,
        }

        media_payload = self._build_media_payload(media_data, modality)
        if media_payload:
            item.metadata.update(media_payload)

        stored_item = MemoryItem(
            id=item.id,
            content=bridge_text,
            memory_type=MemoryType.PERCEPTUAL,
            embedding=item.embedding,
            metadata=item.metadata,
            created_at=item.created_at,
            updated_at=item.updated_at,
            ttl=item.ttl,
            importance=item.importance,
            emotion=item.emotion,
            tags=item.tags,
            location=item.location,
            source=item.source,
        )
        self._items[item.id] = stored_item
        self._item_modalities[item.id] = modality

        payload = self._serialize_item(stored_item)
        self._get_vector_store(modality).store_vector(
            vector_id=stored_item.id,
            vector=stored_item.embedding or [],
            payload=payload,
        )
        self.document_store.store_document(
            doc_id=stored_item.id,
            content=stored_item.content,
            doc_type=self._doc_type(modality),
            metadata=payload["metadata"],
            created_at=stored_item.created_at,
            updated_at=stored_item.updated_at,
        )
        return stored_item.id

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        *,
        query_media_type: str = "text",
        target_media_types: Optional[List[str]] = None,
        cross_modal: bool = True,
    ) -> List[MemoryItem]:
        """默认检索入口。

        说明：
            - 空查询返回全部感知记忆，可按目标模态过滤。
            - 非空查询默认走混合检索，便于 manager 直接复用。
        """
        if limit <= 0:
            return []

        if not query.strip():
            items = self._list_items(target_media_types=target_media_types)
            items.sort(key=lambda item: (item.created_at.timestamp(), item.id), reverse=True)
            return items[:limit]

        return self.search_hybrid(
            query,
            limit=limit,
            query_media_type=query_media_type,
            target_media_types=target_media_types,
            cross_modal=cross_modal,
        )

    def delete(self, memory_id: str) -> bool:
        """删除感知记忆项。"""
        modality = self._item_modalities.get(memory_id)
        success = True

        if modality:
            success = self._get_vector_store(modality).delete(memory_id) and success
        else:
            for store in self._vector_stores.values():
                success = store.delete(memory_id) and success

        success = self.document_store.delete(memory_id) and success
        self._items.pop(memory_id, None)
        self._item_modalities.pop(memory_id, None)
        return success

    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """更新感知记忆项。"""
        current = self._get_item(memory_id)
        if current is None:
            return False

        next_metadata = {**current.metadata, **updates.get("metadata", {})}
        next_modality = self._normalize_media_type(
            updates.get("media_type", next_metadata.get("media_type", "text"))
        )
        updated_item = MemoryItem(
            id=current.id,
            content=updates.get("content", current.content),
            memory_type=MemoryType.PERCEPTUAL,
            embedding=current.embedding,
            metadata=next_metadata,
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
        self.store(
            updated_item,
            media_data=updates.get("media_data"),
            media_type=next_modality,
        )
        return True

    def clear(self) -> bool:
        """清空感知记忆。"""
        success = self.document_store.clear()
        for store in self._vector_stores.values():
            success = store.clear() and success
        self._items.clear()
        self._item_modalities.clear()
        return success

    def store_image(self, image_data: bytes, description: str = "", **metadata) -> str:
        """存储图像记忆。"""
        item = MemoryItem(
            id=self._generate_id(),
            content=description,
            memory_type=MemoryType.PERCEPTUAL,
            metadata=metadata,
            importance=float(metadata.pop("importance", 0.5)),
        )
        return self.store(item, media_data=image_data, media_type="image")

    def store_audio(self, audio_data: bytes, transcript: str = "", **metadata) -> str:
        """存储音频记忆。"""
        item = MemoryItem(
            id=self._generate_id(),
            content=transcript,
            memory_type=MemoryType.PERCEPTUAL,
            metadata=metadata,
            importance=float(metadata.pop("importance", 0.5)),
        )
        return self.store(item, media_data=audio_data, media_type="audio")

    def get_all_items(self, target_media_types: Optional[List[str]] = None) -> List[MemoryItem]:
        """获取全部感知记忆项。"""
        items = self._list_items(target_media_types=target_media_types)
        items.sort(key=lambda item: (item.created_at, item.id))
        return items

    def search_by_keyword(
        self,
        query: str,
        limit: int = 10,
        *,
        query_media_type: str = "text",
        target_media_types: Optional[List[str]] = None,
        cross_modal: bool = True,
    ) -> List[MemoryItem]:
        """关键词搜索。"""
        candidates = self._collect_keyword_candidates(
            query=query,
            limit=limit,
            query_media_type=query_media_type,
            target_media_types=target_media_types,
            cross_modal=cross_modal,
        )
        return self._rank_candidates(query=query, candidates=candidates, limit=limit, use_keyword=True)

    def search_by_semantic(
        self,
        query: str,
        limit: int = 10,
        *,
        query_media_type: str = "text",
        target_media_types: Optional[List[str]] = None,
        cross_modal: bool = True,
    ) -> List[MemoryItem]:
        """语义搜索。"""
        candidates = self._collect_semantic_candidates(
            query=query,
            limit=limit,
            query_media_type=query_media_type,
            target_media_types=target_media_types,
            cross_modal=cross_modal,
        )
        if not candidates:
            return self.search_by_keyword(
                query,
                limit=limit,
                query_media_type=query_media_type,
                target_media_types=target_media_types,
                cross_modal=cross_modal,
            )
        return self._rank_candidates(query=query, candidates=candidates, limit=limit, use_keyword=False)

    def search_hybrid(
        self,
        query: str,
        limit: int = 10,
        *,
        query_media_type: str = "text",
        target_media_types: Optional[List[str]] = None,
        cross_modal: bool = True,
    ) -> List[MemoryItem]:
        """混合搜索。"""
        keyword_candidates = self._collect_keyword_candidates(
            query=query,
            limit=max(limit * 3, limit),
            query_media_type=query_media_type,
            target_media_types=target_media_types,
            cross_modal=cross_modal,
        )
        semantic_candidates = self._collect_semantic_candidates(
            query=query,
            limit=max(limit * 3, limit),
            query_media_type=query_media_type,
            target_media_types=target_media_types,
            cross_modal=cross_modal,
        )
        merged = {item.id: item for item in keyword_candidates}
        for item in semantic_candidates:
            merged[item.id] = item

        if not merged:
            return []

        return self._rank_candidates(query=query, candidates=list(merged.values()), limit=limit, use_keyword=True)

    def score_item(self, query: str, item: MemoryItem) -> Optional[float]:
        """为 manager 提供感知记忆统一评分。"""
        if item.memory_type != MemoryType.PERCEPTUAL:
            return None

        current = self._get_item(item.id) or item
        semantic_similarity = self._semantic_similarity(query, current)
        keyword_similarity = self._keyword_similarity(query, current)
        similarity = max(semantic_similarity, keyword_similarity)
        if similarity <= 0:
            return 0.0
        return self._final_score(current, similarity)

    def _collect_keyword_candidates(
        self,
        query: str,
        limit: int,
        query_media_type: str,
        target_media_types: Optional[List[str]],
        cross_modal: bool,
    ) -> List[MemoryItem]:
        """从文档存储和本地缓存收集关键词候选。"""
        if not query.strip():
            return self._list_items(target_media_types=target_media_types)[:limit]

        modalities = self._resolve_target_modalities(
            query_media_type=query_media_type,
            target_media_types=target_media_types,
            cross_modal=cross_modal,
        )

        candidates: Dict[str, MemoryItem] = {}
        for modality in modalities:
            for document in self.document_store.search_documents_by_metadata_keyword(
                query=query,
                doc_type=self._doc_type(modality),
                limit=max(limit * 3, limit),
            ):
                item = self._document_to_item(document)
                candidates[item.id] = item

        for item in self._list_items(target_media_types=list(modalities)):
            if self._keyword_similarity(query, item) > 0:
                candidates[item.id] = item

        return list(candidates.values())

    def _collect_semantic_candidates(
        self,
        query: str,
        limit: int,
        query_media_type: str,
        target_media_types: Optional[List[str]],
        cross_modal: bool,
    ) -> List[MemoryItem]:
        """从对应模态向量集合收集语义候选。"""
        if not query.strip():
            return self._list_items(target_media_types=target_media_types)[:limit]

        query_embedding = self._embed_text(query)
        if not query_embedding:
            return []

        modalities = self._resolve_target_modalities(
            query_media_type=query_media_type,
            target_media_types=target_media_types,
            cross_modal=cross_modal,
        )

        candidates: Dict[str, MemoryItem] = {}
        for modality in modalities:
            vector_store = self._get_vector_store(modality)
            for result in vector_store.search(query_vector=query_embedding, limit=max(limit * 3, limit)):
                item = self._vector_result_to_item(result, modality)
                if item is not None:
                    candidates[item.id] = item

        return list(candidates.values())

    def _rank_candidates(
        self,
        query: str,
        candidates: Iterable[MemoryItem],
        limit: int,
        use_keyword: bool,
    ) -> List[MemoryItem]:
        """统一候选排序。"""
        scored_items: List[Tuple[float, float, MemoryItem]] = []
        for item in candidates:
            semantic_similarity = self._semantic_similarity(query, item)
            keyword_similarity = self._keyword_similarity(query, item) if use_keyword else 0.0
            similarity = max(semantic_similarity, keyword_similarity)
            if similarity <= 0:
                continue

            score = self._final_score(item, similarity)
            scored_items.append((score, item.created_at.timestamp(), item))

        scored_items.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item for _, _, item in scored_items[:limit]]

    def _final_score(self, item: MemoryItem, similarity: float) -> float:
        """按要求实现统一评分公式。

        说明：
            - 这里把语义检索得到的向量相似度作为主项。
            - 当系统降级到关键词检索时，会用关键词匹配度填充到同一位置，
              这样仍能复用统一排序公式，而不是切换到另一套不可比较的分值体系。
            - 时间近因性采用指数衰减，体现“遗忘曲线”：越新的记忆越容易被召回，
              但旧记忆不会因为年龄稍大就突然失效。
        """
        importance = min(max(item.importance, 0.0), 1.0)
        recency = self._time_recency(item)
        return (similarity * self._SIMILARITY_WEIGHT + recency * self._RECENCY_WEIGHT) * (
            0.8 + importance * 0.4
        )

    def _time_recency(self, item: MemoryItem) -> float:
        """基于遗忘曲线思想计算时间近因性。

        说明：
            - 使用指数衰减 `0.5 ** (age / half_life)`，而不是线性扣分。
            - 这样新近感知内容会自然排前，但较早的重要感知仍保留一部分权重。
            - 当 age 等于 half_life 时，时间权重衰减到 0.5，符合“记忆逐步淡化”而非突变消失。
        """
        age_seconds = max((datetime.now() - item.created_at).total_seconds(), 0.0)
        half_life_seconds = max(self._DEFAULT_HALF_LIFE_HOURS * 3600.0, 1.0)
        return 0.5 ** (age_seconds / half_life_seconds)

    def _semantic_similarity(self, query: str, item: MemoryItem) -> float:
        """计算桥接文本语义相似度。"""
        query_embedding = self._embed_text(query)
        if not query_embedding or not item.embedding:
            return 0.0

        if self.embedding_service is None:
            return 0.0

        return self.embedding_service.cosine_similarity(query_embedding, item.embedding)

    def _keyword_similarity(self, query: str, item: MemoryItem) -> float:
        """计算桥接文本关键词相似度。"""
        raw_query = query.strip().lower()
        item_text = self._item_search_text(item).lower()
        if raw_query and raw_query in item_text:
            return 1.0

        query_tokens = self._tokenize(query)
        item_tokens = self._tokenize(item_text)

        if query_tokens and item_tokens:
            overlap = len(set(query_tokens).intersection(item_tokens))
            return overlap / len(set(query_tokens))

        return 0.0

    def _item_search_text(self, item: MemoryItem) -> str:
        """构造检索文本。"""
        parts = [
            item.content,
            str(item.metadata.get("bridge_text", "")),
            " ".join(str(tag) for tag in item.tags),
            str(item.location or ""),
            str(item.source or ""),
        ]
        return " ".join(part for part in parts if part)

    def _build_bridge_text(
        self,
        content: str,
        media_type: str,
        metadata: Optional[Dict[str, Any]],
        media_data: Optional[Union[bytes, str]],
    ) -> str:
        """构造跨模态桥接文本。"""
        metadata = metadata or {}
        text_parts = [content.strip()]

        # 这里明确使用“描述文本桥接”作为轻量跨模态方案。
        # 若将来接入 CLIP / Whisper / BLIP 等外部依赖，可优先写回 description/transcript/caption，
        # 当前最小实现仍能依靠这些文本字段完成跨模态召回。
        for key in (
            "description",
            "caption",
            "transcript",
            "summary",
            "ocr_text",
            "alt_text",
        ):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())

        processed_media_text = self._process_media(media_data, media_type)
        if processed_media_text:
            text_parts.append(processed_media_text)

        if media_type and media_type != "text":
            text_parts.append(f"模态:{media_type}")

        bridge_text = " ".join(part for part in text_parts if part).strip()
        return bridge_text or content or metadata.get("description", "") or media_type

    def _process_media(self, media_data: Optional[Union[bytes, str]], media_type: str) -> str:
        """处理媒体数据，优雅降级到文本提示。

        说明：
            - 当前仓库没有稳定的多模态推理依赖，这里不强行引入重量级组件。
            - 如果传入的是字符串，优先把它视为外部路径、URL 或人工描述，直接纳入桥接文本。
            - 如果传入的是二进制，则只保留“存在该模态数据”的事实，不尝试做真实解码。
        """
        if isinstance(media_data, str) and media_data.strip():
            return media_data.strip()
        if isinstance(media_data, bytes) and media_data:
            return f"包含{media_type}二进制数据"
        return ""

    def _build_media_payload(
        self,
        media_data: Optional[Union[bytes, str]],
        media_type: str,
    ) -> Dict[str, Any]:
        """构造轻量媒体元数据。"""
        if media_data is None:
            return {}

        if isinstance(media_data, str):
            return {"media_reference": media_data}

        if isinstance(media_data, bytes):
            preview = base64.b64encode(media_data[:256]).decode("utf-8")
            return {
                "media_preview_b64": preview,
                "media_size": len(media_data),
                "media_type": media_type,
            }

        return {}

    def _serialize_item(self, item: MemoryItem) -> Dict[str, Any]:
        """将记忆项序列化为存储载荷。"""
        return {
            "content": item.content,
            "type": item.memory_type.value,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
            "metadata": {
                **(item.metadata or {}),
                "importance": item.importance,
                "emotion": item.emotion,
                "tags": item.tags,
                "location": item.location,
                "source": item.source,
                "ttl": item.ttl,
                "embedding": item.embedding,
            },
        }

    def _vector_result_to_item(
        self,
        result: Dict[str, Any],
        fallback_modality: str,
    ) -> Optional[MemoryItem]:
        """把向量检索结果转回 MemoryItem。"""
        memory_id = str(result.get("id", ""))
        if not memory_id:
            return None
        existing = self._get_item(memory_id)
        if existing is not None:
            return existing

        payload = result.get("payload", {}) or {}
        metadata = payload.get("metadata", {}) or {}
        item = MemoryItem(
            id=memory_id,
            content=payload.get("content", ""),
            memory_type=MemoryType.PERCEPTUAL,
            embedding=result.get("vector", []),
            metadata=metadata,
            created_at=self._parse_datetime(payload.get("created_at")),
            updated_at=self._parse_datetime(payload.get("updated_at")),
            ttl=metadata.get("ttl"),
            importance=float(metadata.get("importance", 0.5)),
            emotion=metadata.get("emotion"),
            tags=list(metadata.get("tags", [])),
            location=metadata.get("location"),
            source=metadata.get("source"),
        )
        modality = self._normalize_media_type(metadata.get("media_type", fallback_modality))
        self._items[item.id] = item
        self._item_modalities[item.id] = modality
        return item

    def _document_to_item(self, document: Dict[str, Any]) -> MemoryItem:
        """把文档存储结果转回 MemoryItem。"""
        metadata = document.get("metadata", {}) or {}
        bridge_text = str(metadata.get("bridge_text", document.get("content", "")))
        embedding = metadata.get("embedding")
        if not embedding:
            embedding = self._embed_text(bridge_text)
        item = MemoryItem(
            id=document.get("id", ""),
            content=document.get("content", ""),
            memory_type=MemoryType.PERCEPTUAL,
            embedding=embedding,
            metadata=metadata,
            created_at=self._parse_datetime(document.get("created_at")),
            updated_at=self._parse_datetime(document.get("updated_at")),
            ttl=metadata.get("ttl"),
            importance=float(metadata.get("importance", 0.5)),
            emotion=metadata.get("emotion"),
            tags=list(metadata.get("tags", [])),
            location=metadata.get("location"),
            source=metadata.get("source"),
        )
        self._items[item.id] = item
        self._item_modalities[item.id] = self._normalize_media_type(metadata.get("media_type", "text"))
        return item

    def _get_item(self, memory_id: str) -> Optional[MemoryItem]:
        """优先从内存缓存获取记忆项。"""
        item = self._items.get(memory_id)
        if item is not None:
            return item

        document = self.document_store.get_document(memory_id)
        if document is None:
            return None
        return self._document_to_item(document)

    def _list_items(self, target_media_types: Optional[List[str]] = None) -> List[MemoryItem]:
        """列出感知记忆项，并支持模态过滤。"""
        modalities = self._resolve_target_modalities(
            query_media_type="text",
            target_media_types=target_media_types,
            cross_modal=True,
        )
        items: Dict[str, MemoryItem] = {}

        for item in self._items.values():
            if self._normalize_media_type(item.metadata.get("media_type", "text")) in modalities:
                items[item.id] = item

        for modality in modalities:
            for document in self.document_store.list_documents(doc_type=self._doc_type(modality), limit=1000):
                items[document["id"]] = self._document_to_item(document)

        return list(items.values())

    def _resolve_target_modalities(
        self,
        query_media_type: str,
        target_media_types: Optional[List[str]],
        cross_modal: bool,
    ) -> List[str]:
        """解析本次检索要命中的模态集合。"""
        normalized_query_media_type = self._normalize_media_type(query_media_type)
        if target_media_types:
            return [self._normalize_media_type(modality) for modality in target_media_types]
        if cross_modal:
            discovered = {
                self._normalize_media_type(item.metadata.get("media_type", "text"))
                for item in self._items.values()
            }
            if discovered:
                return sorted(discovered)
            return [normalized_query_media_type]
        return [normalized_query_media_type]

    def _get_vector_store(self, media_type: str) -> QdrantStore:
        """按模态惰性创建独立向量集合。"""
        modality = self._normalize_media_type(media_type)
        store = self._vector_stores.get(modality)
        if store is not None:
            return store

        collection_name = f"{self.config.vector_collection_name}_perceptual_{modality}"
        if hasattr(self.config, "model_copy"):
            scoped_config = self.config.model_copy(update={"vector_collection_name": collection_name})
        else:
            scoped_config = self.config.copy(update={"vector_collection_name": collection_name})

        store = QdrantStore(scoped_config)
        self._vector_stores[modality] = store
        return store

    def _embed_text(self, text: str) -> List[float]:
        """安全地生成文本嵌入。"""
        if self.embedding_service is None or not text.strip():
            return []
        try:
            return self.embedding_service.embed(text)
        except Exception:
            return []

    def _doc_type(self, media_type: str) -> str:
        """返回文档类型名。"""
        return f"perceptual_{self._normalize_media_type(media_type)}"

    def _normalize_media_type(self, media_type: Optional[str]) -> str:
        """标准化模态名称。"""
        normalized = (media_type or "text").strip().lower() or "text"
        return normalized if normalized in self._SUPPORTED_MODALITIES else normalized

    def _tokenize(self, text: str) -> List[str]:
        """对中英文做轻量分词。"""
        text = text.lower()
        tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
        return [token for token in tokens if token.strip()]

    def _parse_datetime(self, value: Any) -> datetime:
        """解析存储中的时间字符串。"""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.now()

    def _generate_id(self) -> str:
        """生成唯一 ID。"""
        return str(uuid.uuid4())
