"""Embedding services for the experimental memory subsystem."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import List
from urllib import error, request

import numpy as np

from src.memory.base import MemoryConfig


class EmbeddingService(ABC):
    """Abstract interface for text embeddings."""

    def __init__(self, config: MemoryConfig):
        self.config = config

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed one text."""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed many texts."""

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2:
            return 0.0

        vec1_np = np.array(vec1, dtype=float)
        vec2_np = np.array(vec2, dtype=float)
        if len(vec1_np) != len(vec2_np):
            min_len = min(len(vec1_np), len(vec2_np))
            vec1_np = vec1_np[:min_len]
            vec2_np = vec2_np[:min_len]

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))


class GiteeEmbedding(EmbeddingService):
    """Legacy raw HTTP embedding client."""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.api_key = config.embedding_api_key
        self.base_url = config.embedding_base_url

    def embed(self, text: str) -> List[float]:
        return self._request_embeddings(text)[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self._request_embeddings(texts)

    def _request_embeddings(self, inputs: str | List[str]) -> List[List[float]]:
        payload = {
            "model": self.config.embedding_model,
            "input": inputs,
            "encoding_format": "float",
        }
        if self.config.embedding_dimension > 0:
            payload["dimensions"] = self.config.embedding_dimension

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        http_request = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"embedding request failed: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"embedding request failed: {exc.reason}") from exc

        data = body.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"unexpected embedding response: {body}")

        embeddings: List[List[float]] = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                raise RuntimeError(f"unexpected embedding item: {item}")
            embeddings.append(embedding)
        return embeddings


class DashScopeEmbedding(GiteeEmbedding):
    """Compatibility alias for older imports."""


class LocalEmbedding(EmbeddingService):
    """Placeholder local implementation."""

    def embed(self, text: str) -> List[float]:
        del text
        return [0.0] * self.config.embedding_dimension

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


class OpenAISDKEmbedding(EmbeddingService):
    """OpenAI SDK based embedding client for OpenAI-compatible providers."""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("missing dependency: openai") from exc

        self._openai_cls = OpenAI
        self.client = None
        self.model = config.embedding_model
        self.dimension = config.embedding_dimension

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.config.embedding_api_key:
            raise RuntimeError("missing embedding api key")

        if self.client is None:
            self.client = self._openai_cls(
                base_url=self.config.embedding_base_url or "https://api.openai.com/v1",
                api_key=self.config.embedding_api_key,
            )

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimension if self.dimension > 0 else None,
                encoding_format="float",
            )
        except Exception as exc:
            raise RuntimeError(f"openai sdk embedding request failed: {exc}") from exc

        return [data.embedding for data in response.data]


class TFIDFEmbedding(EmbeddingService):
    """Local fallback embedding without remote dependencies."""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=config.embedding_dimension)
        self.vocabulary_fitted = False

    def embed(self, text: str) -> List[float]:
        if not self.vocabulary_fitted:
            self.vectorizer.fit([text])
            self.vocabulary_fitted = True
        vector = self.vectorizer.transform([text])
        return vector.toarray()[0].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.vocabulary_fitted:
            self.vectorizer.fit(texts)
            self.vocabulary_fitted = True
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray().tolist()


def create_embedding_service(config: MemoryConfig) -> EmbeddingService:
    """Create the configured embedding service."""
    provider = (config.embedding_provider or "").lower()
    if provider == "tfidf":
        return TFIDFEmbedding(config)
    if provider == "local":
        return LocalEmbedding(config)
    if provider == "gitee":
        return GiteeEmbedding(config)
    return OpenAISDKEmbedding(config)
