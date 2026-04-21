"""记忆系统基础数据结构"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, List, Dict
from pydantic import BaseModel


class MemoryType(str, Enum):
    """记忆类型枚举"""
    WORKING = "working"      # 工作记忆
    EPISODIC = "episodic"    # 情景记忆
    SEMANTIC = "semantic"    # 语义记忆
    PERCEPTUAL = "perceptual"  # 感知记忆


@dataclass
class MemoryItem:
    """记忆项基础数据结构"""
    
    id: str
    content: str
    memory_type: MemoryType
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    ttl: Optional[int] = None  # 生存时间（秒），None表示永久


class MemoryConfig(BaseModel):
    """记忆系统配置"""
    
    # 嵌入配置
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    
    # 向量存储配置
    vector_store_url: str = "http://localhost:6333"
    vector_collection_name: str = "memories"
    
    # 图存储配置
    graph_store_url: str = "bolt://localhost:7687"
    graph_username: str = "neo4j"
    graph_password: str = "password"
    
    # 文档存储配置
    document_store_path: str = "data/memories.db"
    
    # 记忆配置
    working_memory_ttl: int = 300  # 工作记忆生存时间（秒）
    max_working_memory_items: int = 100  # 工作记忆最大项数


class BaseMemory(ABC):
    """记忆系统基础抽象类"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
    
    @abstractmethod
    def store(self, item: MemoryItem) -> str:
        """存储记忆项"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """检索相关记忆项"""
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """删除记忆项"""
        pass
    
    @abstractmethod
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """更新记忆项"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """清空所有记忆"""
        pass