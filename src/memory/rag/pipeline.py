"""实验性记忆子系统的RAG管道实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.memory.base import MemoryConfig
from src.memory.embedding import EmbeddingService, create_embedding_service
from src.memory.rag.document import DocumentProcessor


class RAGPipeline(ABC):
    """RAG管道的抽象基类。"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        """初始化RAG管道。
        
        参数:
            config: 记忆配置，默认为从环境变量读取的配置
        """
        self.config = config or MemoryConfig.from_env()
        self.embedding_service: EmbeddingService = create_embedding_service(self.config)
        self.document_processor = DocumentProcessor()

    @abstractmethod
    def ingest(self, documents: List[Dict[str, Any]]) -> List[str]:
        """将文档摄入管道。
        
        参数:
            documents: 文档列表
            
        返回:
            文档ID列表
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关文档。
        
        参数:
            query: 查询文本
            top_k: 返回结果的最大数量
            
        返回:
            相关文档列表
        """

    @abstractmethod
    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """生成答案。
        
        参数:
            query: 查询文本
            context: 上下文文档列表
            
        返回:
            生成的答案文本
        """

    def query(self, query: str) -> str:
        """执行完整查询流程。
        
        参数:
            query: 查询文本
            
        返回:
            生成的答案文本
        """
        context = self.retrieve(query)
        answer = self.generate(query, context)
        return answer


class SimpleRAGPipeline(RAGPipeline):
    """仅使用向量的简单RAG管道。"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        """初始化简单RAG管道。
        
        参数:
            config: 记忆配置，默认为从环境变量读取的配置
        """
        super().__init__(config)
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []

    def ingest(self, documents: List[Dict[str, Any]]) -> List[str]:
        """将文档摄入管道。
        
        参数:
            documents: 文档列表
            
        返回:
            文档ID列表
        """
        doc_ids: List[str] = []

        for doc in documents:
            processed = self.document_processor.process(doc)
            embedding = self.embedding_service.embed(processed["content"])

            doc_id = f"doc_{len(self.documents)}"
            self.documents.append(
                {
                    "id": doc_id,
                    "content": processed["content"],
                    "metadata": processed["metadata"],
                    "embedding": embedding,
                }
            )
            self.embeddings.append(embedding)
            doc_ids.append(doc_id)

        return doc_ids

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关文档。
        
        参数:
            query: 查询文本
            top_k: 返回结果的最大数量
            
        返回:
            相关文档列表
        """
        if not self.documents:
            return []

        query_embedding = self.embedding_service.embed(query)

        similarities = []
        for idx, doc_embedding in enumerate(self.embeddings):
            similarity = self.embedding_service.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((idx, similarity))

        results = []
        similarities.sort(key=lambda item: item[1], reverse=True)
        for idx, similarity in similarities[:top_k]:
            doc = self.documents[idx]
            results.append(
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity": similarity,
                }
            )

        return results

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """生成答案。
        
        参数:
            query: 查询文本
            context: 上下文文档列表
            
        返回:
            生成的答案文本
        """
        if not context:
            return "未找到相关信息。"

        context_text = "\n\n".join(
            [f"[文档 {idx + 1}]\n{doc['content'][:500]}..." for idx, doc in enumerate(context)]
        )
        return f"问题: {query}\n\n上下文:\n{context_text[:1000]}..."

    def clear(self):
        """清空管道中的所有文档和嵌入。"""
        self.documents.clear()
        self.embeddings.clear()


class HybridRAGPipeline(RAGPipeline):
    """混合RAG管道：结合向量和关键词检索。"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        """初始化混合RAG管道。
        
        参数:
            config: 记忆配置，默认为从环境变量读取的配置
        """
        super().__init__(config)
        self.vector_docs: List[Dict[str, Any]] = []
        self.vector_embeddings: List[List[float]] = []
        self.keyword_index: Dict[str, List[int]] = {}

    def ingest(self, documents: List[Dict[str, Any]]) -> List[str]:
        """将文档摄入管道。
        
        参数:
            documents: 文档列表
            
        返回:
            文档ID列表
        """
        doc_ids: List[str] = []

        for doc in documents:
            processed = self.document_processor.process(doc)
            embedding = self.embedding_service.embed(processed["content"])

            doc_id = f"doc_{len(self.vector_docs)}"
            doc_idx = len(self.vector_docs)
            self.vector_docs.append(
                {
                    "id": doc_id,
                    "content": processed["content"],
                    "metadata": processed["metadata"],
                    "embedding": embedding,
                }
            )
            self.vector_embeddings.append(embedding)
            self._build_keyword_index(doc_idx, processed["content"])
            doc_ids.append(doc_id)

        return doc_ids

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关文档。
        
        参数:
            query: 查询文本
            top_k: 返回结果的最大数量
            
        返回:
            相关文档列表
        """
        if not self.vector_docs:
            return []

        vector_results = self._vector_retrieve(query, top_k)
        keyword_results = self._keyword_retrieve(query, top_k)
        return self._merge_results(vector_results, keyword_results, top_k)

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """生成答案。
        
        参数:
            query: 查询文本
            context: 上下文文档列表
            
        返回:
            生成的答案文本
        """
        return SimpleRAGPipeline.generate(self, query, context)

    def _build_keyword_index(self, doc_idx: int, content: str):
        """构建关键词索引。
        
        参数:
            doc_idx: 文档索引
            content: 文档内容
        """
        for word in content.lower().split():
            if len(word) <= 2:
                continue
            bucket = self.keyword_index.setdefault(word, [])
            if doc_idx not in bucket:
                bucket.append(doc_idx)

    def _vector_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """向量检索。
        
        参数:
            query: 查询文本
            top_k: 返回结果的最大数量
            
        返回:
            向量检索结果列表
        """
        query_embedding = self.embedding_service.embed(query)

        similarities = []
        for idx, doc_embedding in enumerate(self.vector_embeddings):
            similarity = self.embedding_service.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((idx, similarity))

        similarities.sort(key=lambda item: item[1], reverse=True)

        results = []
        for idx, similarity in similarities[:top_k]:
            doc = self.vector_docs[idx]
            results.append(
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity": similarity,
                    "retrieval_method": "vector",
                }
            )

        return results

    def _keyword_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """关键词检索。
        
        参数:
            query: 查询文本
            top_k: 返回结果的最大数量
            
        返回:
            关键词检索结果列表
        """
        query_words = query.lower().split()
        doc_scores: Dict[int, int] = {}

        for word in query_words:
            for doc_idx in self.keyword_index.get(word, []):
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 1

        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_idx, score in sorted_docs[:top_k]:
            doc = self.vector_docs[doc_idx]
            results.append(
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity": score / len(query_words) if query_words else 0.0,
                    "retrieval_method": "keyword",
                }
            )

        return results

    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """合并向量和关键词检索结果。
        
        参数:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            top_k: 返回结果的最大数量
            
        返回:
            合并后的结果列表
        """
        seen_contents = set()
        merged = []

        for result in vector_results:
            content_hash = hash(result["content"][:100])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged.append(result)

        for result in keyword_results:
            if len(merged) >= top_k:
                break

            content_hash = hash(result["content"][:100])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged.append(result)

        return merged[:top_k]
