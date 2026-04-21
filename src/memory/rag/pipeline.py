"""RAG管道实现"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from src.memory.base import MemoryConfig
from src.memory.embedding import EmbeddingService
from src.memory.rag.document import DocumentProcessor


class RAGPipeline(ABC):
    """RAG管道抽象类"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        # 使用TFIDF嵌入作为默认实现
        from src.memory.embedding import TFIDFEmbedding
        self.embedding_service = TFIDFEmbedding(self.config)
        self.document_processor = DocumentProcessor()
    
    @abstractmethod
    def ingest(self, documents: List[Dict[str, Any]]) -> List[str]:
        """文档摄取"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关文档"""
        pass
    
    @abstractmethod
    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """生成答案"""
        pass
    
    def query(self, query: str) -> str:
        """端到端查询"""
        # 1. 检索
        context = self.retrieve(query)
        
        # 2. 生成
        answer = self.generate(query, context)
        
        return answer


class SimpleRAGPipeline(RAGPipeline):
    """简单RAG管道实现"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        super().__init__(config)
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
    
    def ingest(self, documents: List[Dict[str, Any]]) -> List[str]:
        """文档摄取"""
        doc_ids = []
        
        for doc in documents:
            # 处理文档
            processed = self.document_processor.process(doc)
            
            # 生成嵌入
            embedding = self.embedding_service.embed(processed["content"])
            
            # 存储
            doc_id = f"doc_{len(self.documents)}"
            self.documents.append({
                "id": doc_id,
                "content": processed["content"],
                "metadata": processed["metadata"],
                "embedding": embedding
            })
            self.embeddings.append(embedding)
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关文档"""
        if not self.documents:
            return []
        
        # 生成查询嵌入
        query_embedding = self.embedding_service.embed(query)
        
        # 计算相似度
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.embedding_service.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # 排序并取top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_k]]
        
        # 返回文档信息
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity": similarities[idx][1]
            })
        
        return results
    
    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """生成答案（简化实现）"""
        if not context:
            return "未找到相关信息。"
        
        # 构建上下文
        context_text = "\n\n".join([
            f"[文档 {i+1}]\n{doc['content'][:500]}..."
            for i, doc in enumerate(context)
        ])
        
        # 这里应该调用LLM生成答案
        # 简化实现：返回上下文摘要
        
        return f"基于以下信息回答：{query}\n\n相关上下文：\n{context_text[:1000]}..."
    
    def clear(self):
        """清空管道"""
        self.documents.clear()
        self.embeddings.clear()


class HybridRAGPipeline(RAGPipeline):
    """混合RAG管道（结合向量检索和关键词检索）"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        super().__init__(config)
        self.vector_docs: List[Dict[str, Any]] = []
        self.vector_embeddings: List[List[float]] = []
        self.keyword_index: Dict[str, List[int]] = {}
    
    def ingest(self, documents: List[Dict[str, Any]]) -> List[str]:
        """文档摄取（建立向量和关键词索引）"""
        doc_ids = []
        
        for doc_idx, doc in enumerate(documents):
            # 处理文档
            processed = self.document_processor.process(doc)
            
            # 生成嵌入
            embedding = self.embedding_service.embed(processed["content"])
            
            # 存储向量
            doc_id = f"doc_{len(self.vector_docs)}"
            self.vector_docs.append({
                "id": doc_id,
                "content": processed["content"],
                "metadata": processed["metadata"],
                "embedding": embedding
            })
            self.vector_embeddings.append(embedding)
            
            # 建立关键词索引
            self._build_keyword_index(doc_idx, processed["content"])
            
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """混合检索"""
        if not self.vector_docs:
            return []
        
        # 向量检索
        vector_results = self._vector_retrieve(query, top_k)
        
        # 关键词检索
        keyword_results = self._keyword_retrieve(query, top_k)
        
        # 合并结果（去重，加权）
        combined = self._merge_results(vector_results, keyword_results, top_k)
        
        return combined
    
    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """生成答案"""
        return SimpleRAGPipeline.generate(self, query, context)
    
    def _build_keyword_index(self, doc_idx: int, content: str):
        """建立关键词索引"""
        # 简单分词（实际应该使用更好的分词器）
        words = content.lower().split()
        
        for word in words:
            if len(word) > 2:  # 忽略短词
                if word not in self.keyword_index:
                    self.keyword_index[word] = []
                if doc_idx not in self.keyword_index[word]:
                    self.keyword_index[word].append(doc_idx)
    
    def _vector_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """向量检索"""
        query_embedding = self.embedding_service.embed(query)
        
        similarities = []
        for i, doc_embedding in enumerate(self.vector_embeddings):
            similarity = self.embedding_service.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, similarity in similarities[:top_k]:
            doc = self.vector_docs[idx]
            results.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity": similarity,
                "retrieval_method": "vector"
            })
        
        return results
    
    def _keyword_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """关键词检索"""
        query_words = query.lower().split()
        doc_scores = {}
        
        for word in query_words:
            if word in self.keyword_index:
                for doc_idx in self.keyword_index[word]:
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 1
        
        # 按分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_idx, score in sorted_docs[:top_k]:
            doc = self.vector_docs[doc_idx]
            results.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity": score / len(query_words),  # 归一化
                "retrieval_method": "keyword"
            })
        
        return results
    
    def _merge_results(self, vector_results: List[Dict], keyword_results: List[Dict], top_k: int) -> List[Dict]:
        """合并检索结果"""
        # 使用集合去重（基于内容）
        seen_contents = set()
        merged = []
        
        # 优先向量检索结果
        for result in vector_results:
            content_hash = hash(result["content"][:100])  # 简单哈希
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged.append(result)
        
        # 补充关键词检索结果
        for result in keyword_results:
            if len(merged) >= top_k:
                break
            
            content_hash = hash(result["content"][:100])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged.append(result)
        
        return merged[:top_k]