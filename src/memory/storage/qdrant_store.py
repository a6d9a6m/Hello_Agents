"""Qdrant向量存储实现"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np

from src.memory.base import MemoryConfig


class QdrantStore:
    """Qdrant向量存储客户端"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.client = None
        self.collection_name = config.vector_collection_name if config else "memories"
        
        # 延迟初始化客户端
        self._initialized = False
    
    def _ensure_initialized(self):
        """确保客户端已初始化"""
        if not self._initialized:
            try:
                # 尝试导入qdrant-client
                import qdrant_client
                from qdrant_client.http import models
                
                self.client = qdrant_client.QdrantClient(
                    url=self.config.vector_store_url,
                    timeout=30
                )
                
                # 检查集合是否存在，不存在则创建
                collections = self.client.get_collections().collections
                collection_names = [col.name for col in collections]
                
                if self.collection_name not in collection_names:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=self.config.embedding_dimension,
                            distance=models.Distance.COSINE
                        )
                    )
                
                self._initialized = True
                
            except ImportError:
                print("警告：未安装qdrant-client，使用模拟模式")
                self.client = None
                self._initialized = True
    
    def store_vector(self, vector_id: str, vector: List[float], payload: Dict[str, Any]) -> bool:
        """存储向量"""
        self._ensure_initialized()
        
        if self.client is None:
            # 模拟模式
            return True
        
        try:
            from qdrant_client.http import models
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=vector_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"存储向量失败：{e}")
            return False
    
    def search(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        self._ensure_initialized()
        
        if self.client is None:
            # 模拟模式：返回空结果
            return []
        
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            return [
                {
                    "id": result.id,
                    "vector": result.vector,
                    "payload": result.payload,
                    "score": result.score
                }
                for result in results
            ]
        except Exception as e:
            print(f"搜索向量失败：{e}")
            return []
    
    def delete(self, vector_id: str) -> bool:
        """删除向量"""
        self._ensure_initialized()
        
        if self.client is None:
            return True
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[vector_id]
            )
            return True
        except Exception as e:
            print(f"删除向量失败：{e}")
            return False
    
    def clear(self) -> bool:
        """清空集合"""
        self._ensure_initialized()
        
        if self.client is None:
            return True
        
        try:
            # 删除并重新创建集合
            self.client.delete_collection(self.collection_name)
            from qdrant_client.http import models
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.embedding_dimension,
                    distance=models.Distance.COSINE
                )
            )
            return True
        except Exception as e:
            print(f"清空集合失败：{e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        self._ensure_initialized()
        
        if self.client is None:
            return {"status": "simulated", "count": 0}
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "status": "ok",
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": len(collection_info.segments)
            }
        except Exception as e:
            print(f"获取统计信息失败：{e}")
            return {"status": "error", "error": str(e)}