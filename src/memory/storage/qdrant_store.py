"""Qdrant vector store with local fallback cache."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from src.memory.base import MemoryConfig


class QdrantStore:
    """Thin Qdrant wrapper used by memory backends."""

    _DISTANCE_MAP = {
        "cosine": "COSINE",
        "dot": "DOT",
        "euclid": "EUCLID",
        "manhattan": "MANHATTAN",
    }

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig.from_env()
        self.client = None
        self.collection_name = self.config.vector_collection_name
        self._local_points: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return

        try:
            import qdrant_client
            from qdrant_client.http import models

            distance_name = self._DISTANCE_MAP.get(self.config.vector_distance.lower(), "COSINE")
            distance = getattr(models.Distance, distance_name)

            self.client = qdrant_client.QdrantClient(
                url=self.config.vector_store_url,
                api_key=self.config.vector_store_api_key,
                timeout=self.config.vector_timeout,
            )

            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.config.embedding_dimension,
                        distance=distance,
                    ),
                )
        except ImportError:
            self.client = None
        except Exception as exc:
            print(f"warning: qdrant unavailable, fallback to local cache: {exc}")
            self.client = None
        finally:
            self._initialized = True

    def is_available(self) -> bool:
        self._ensure_initialized()
        return self.client is not None

    def store_vector(self, vector_id: str, vector: List[float], payload: Dict[str, Any]) -> bool:
        self._ensure_initialized()
        normalized_id = self._normalize_point_id(vector_id)
        stored_payload = self._with_internal_payload(vector_id=vector_id, payload=payload)
        self._local_points[str(vector_id)] = {
            "id": str(vector_id),
            "vector": list(vector or []),
            "payload": stored_payload,
        }

        if self.client is None:
            return True

        try:
            from qdrant_client.http import models

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=normalized_id,
                        vector=list(vector or []),
                        payload=stored_payload,
                    )
                ],
            )
            return True
        except Exception as exc:
            print(f"store vector failed: {exc}")
            return False

    def search(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        if self.client is None:
            return self._search_local(query_vector=query_vector, limit=limit)

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=list(query_vector or []),
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
            results = getattr(response, "points", response)
            normalized = []
            for result in results:
                payload = result.payload or {}
                normalized.append(
                    {
                        "id": str(payload.get("_memory_id") or result.id),
                        "vector": getattr(result, "vector", None),
                        "payload": payload,
                        "score": float(result.score or 0.0),
                    }
                )
            return normalized
        except Exception as exc:
            print(f"search vector failed: {exc}")
            return self._search_local(query_vector=query_vector, limit=limit)

    def delete(self, vector_id: str) -> bool:
        self._ensure_initialized()
        self._local_points.pop(str(vector_id), None)
        if self.client is None:
            return True

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[self._normalize_point_id(vector_id)],
            )
            return True
        except Exception as exc:
            print(f"delete vector failed: {exc}")
            return False

    def clear(self) -> bool:
        self._ensure_initialized()
        self._local_points.clear()
        if self.client is None:
            return True

        try:
            self.client.delete_collection(self.collection_name)
            self._initialized = False
            self._ensure_initialized()
            return True
        except Exception as exc:
            print(f"clear collection failed: {exc}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        self._ensure_initialized()
        if self.client is None:
            return {"status": "simulated", "count": len(self._local_points)}

        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "status": "ok",
                "vectors_count": getattr(collection_info, "vectors_count", None),
                "points_count": getattr(collection_info, "points_count", None),
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def get_all_points(self) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        return list(self._local_points.values())

    def _search_local(self, query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []

        scored_results = []
        query = np.array(query_vector or [], dtype=float)
        query_norm = float(np.linalg.norm(query)) if query.size else 0.0

        for point in self._local_points.values():
            vector = np.array(point.get("vector") or [], dtype=float)
            vector_norm = float(np.linalg.norm(vector)) if vector.size else 0.0
            if query_norm == 0.0 or vector_norm == 0.0:
                score = 0.0
            else:
                score = float(np.dot(query, vector) / (query_norm * vector_norm))
            scored_results.append(
                {
                    "id": point.get("id", ""),
                    "vector": point.get("vector", []),
                    "payload": point.get("payload", {}),
                    "score": score,
                }
            )

        scored_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return scored_results[:limit]

    def _normalize_point_id(self, vector_id: str) -> int | str:
        raw = str(vector_id)
        if raw.isdigit():
            return int(raw)
        try:
            return str(uuid.UUID(raw))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

    def _with_internal_payload(self, vector_id: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = dict(payload or {})
        normalized["_memory_id"] = str(vector_id)
        return normalized
