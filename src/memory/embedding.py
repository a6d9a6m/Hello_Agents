"""统一嵌入服务"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from src.memory.base import MemoryConfig


class EmbeddingService(ABC):
    """嵌入服务抽象类"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """将文本转换为嵌入向量"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量转换文本为嵌入向量"""
        pass
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # 确保向量维度一致
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


class DashScopeEmbedding(EmbeddingService):
    """阿里云DashScope嵌入服务"""
    
    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        # 初始化DashScope客户端
        self.api_key = config.embedding_model  # 这里假设配置中存储了API key
        # 实际实现需要安装dashscope包
    
    def embed(self, text: str) -> List[float]:
        """使用DashScope生成嵌入"""
        # 实际实现
        # import dashscope
        # dashscope.api_key = self.api_key
        # response = dashscope.TextEmbedding.call(
        #     model=self.config.embedding_model,
        #     input=text
        # )
        # return response.output.embeddings[0].embedding
        
        # 临时返回随机向量
        return [0.0] * self.config.embedding_dimension
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入"""
        return [self.embed(text) for text in texts]


class LocalEmbedding(EmbeddingService):
    """本地嵌入模型（如SentenceTransformers）"""
    
    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        # 加载本地模型
        # 实际实现需要安装sentence-transformers包
        self.model = None
    
    def embed(self, text: str) -> List[float]:
        """使用本地模型生成嵌入"""
        # 实际实现
        # if self.model is None:
        #     from sentence_transformers import SentenceTransformer
        #     self.model = SentenceTransformer(self.config.embedding_model)
        # return self.model.encode(text).tolist()
        
        # 临时返回随机向量
        return [0.0] * self.config.embedding_dimension
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入"""
        # 实际实现
        # if self.model is None:
        #     from sentence_transformers import SentenceTransformer
        #     self.model = SentenceTransformer(self.config.embedding_model)
        # embeddings = self.model.encode(texts)
        # return [emb.tolist() for emb in embeddings]
        
        return [self.embed(text) for text in texts]


class TFIDFEmbedding(EmbeddingService):
    """TF-IDF嵌入（轻量级，无需外部依赖）"""
    
    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=config.embedding_dimension)
        self.vocabulary_fitted = False
    
    def embed(self, text: str) -> List[float]:
        """使用TF-IDF生成嵌入"""
        if not self.vocabulary_fitted:
            # 需要先拟合词汇表
            self.vectorizer.fit([text])
            self.vocabulary_fitted = True
        
        vector = self.vectorizer.transform([text])
        return vector.toarray()[0].tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入"""
        if not self.vocabulary_fitted:
            self.vectorizer.fit(texts)
            self.vocabulary_fitted = True
        
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray().tolist()