"""实验性记忆子系统的核心类型定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


@lru_cache(maxsize=1)
def _read_dotenv() -> Dict[str, str]:
    """读取 .env 文件中的环境变量，使用缓存提高性能。
    
    返回:
        环境变量键值对字典
    """
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return {}

    values: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value

    return values


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """获取环境变量值，优先从系统环境变量获取，其次从 .env 文件获取。
    
    参数:
        name: 环境变量名
        default: 默认值
        
    返回:
        环境变量值或默认值
    """
    value = os.getenv(name)
    if value not in (None, ""):
        return value
    return _read_dotenv().get(name, default)


def _env_int(name: str, default: int) -> int:
    """获取整数类型的环境变量值。
    
    参数:
        name: 环境变量名
        default: 默认整数值
        
    返回:
        转换后的整数值或默认值
    """
    value = _env(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


class MemoryType(str, Enum):
    """支持的记忆类型枚举。
    
    工作记忆: 短期记忆，用于当前任务
    情景记忆: 事件和经历的记忆
    语义记忆: 事实和知识的记忆
    感知记忆: 感官信息的记忆
    """

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PERCEPTUAL = "perceptual"


@dataclass
class MemoryItem:
    """单个记忆记录的数据类。
    
    属性:
        id: 唯一标识符
        content: 记忆内容文本
        memory_type: 记忆类型
        embedding: 文本嵌入向量
        metadata: 额外元数据字典
        created_at: 创建时间
        updated_at: 更新时间
        ttl: 生存时间（秒）
        importance: 重要性评分（0.0-1.0）
        emotion: 情感标签
        tags: 标签列表
        location: 位置信息
        source: 来源信息
    """

    id: str
    content: str
    memory_type: MemoryType
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    ttl: Optional[int] = None
    importance: float = 0.5
    emotion: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    location: Optional[str] = None
    source: Optional[str] = None


class MemoryConfig(BaseModel):
    """记忆子系统的配置类。
    
    属性:
        embedding_provider: 嵌入服务提供商
        embedding_model: 嵌入模型名称
        embedding_api_key: API密钥
        embedding_base_url: 嵌入服务基础URL
        embedding_user: 用户标识
        embedding_dimension: 嵌入向量维度
        vector_store_url: 向量存储URL
        vector_collection_name: 向量集合名称
        graph_store_url: 图数据库URL
        graph_username: 图数据库用户名
        graph_password: 图数据库密码
        document_store_path: 文档存储路径
        working_memory_ttl: 工作记忆生存时间（秒）
        max_working_memory_items: 最大工作记忆项数
    """

    embedding_provider: str = "openai"
    embedding_model: str = "Qwen3-Embedding-0.6B"
    embedding_api_key: Optional[str] = None
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_dimension: int = 384

    vector_store_url: str = "http://localhost:6333"
    vector_collection_name: str = "memories"
    vector_store_api_key: Optional[str] = None
    vector_distance: str = "cosine"
    vector_timeout: int = 30

    graph_store_url: str = "bolt://localhost:7687"
    graph_username: str = "neo4j"
    graph_password: str = "password"
    graph_database: str = "neo4j"
    graph_connection_timeout: int = 60
    graph_max_connection_lifetime: int = 3600
    graph_max_connection_pool_size: int = 50

    document_store_path: str = "data/memories.db"

    working_memory_ttl: int = 300
    max_working_memory_items: int = 100

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """从环境变量创建配置实例。
        
        返回:
            配置好的MemoryConfig实例
        """
        llm_base_url = _env("LLM_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1"
        embed_base_url = _env("EMBED_BASE_URL")
        embed_api_key = _env("EMBED_API_KEY")
        if not embed_api_key and (embed_base_url in (None, "", llm_base_url)):
            embed_api_key = _env("LLM_API_KEY")

        return cls(
            embedding_provider=(_env("EMBED_PROVIDER", _env("EMBED_MODEL_TYPE", "openai")) or "openai").lower(),
            embedding_model=_env("EMBED_MODEL", _env("EMBED_MODEL_NAME", "Qwen3-Embedding-0.6B"))
            or "Qwen3-Embedding-0.6B",
            embedding_api_key=embed_api_key,
            embedding_base_url=embed_base_url or llm_base_url,
            embedding_dimension=_env_int(
                "EMBED_DIMENSIONS",
                _env_int("QDRANT_VECTOR_SIZE", 384),
            ),
            vector_store_url=_env("QDRANT_URL", "http://localhost:6333")
            or "http://localhost:6333",
            vector_collection_name=_env("QDRANT_COLLECTION", "memories") or "memories",
            vector_store_api_key=_env("QDRANT_API_KEY"),
            vector_distance=(_env("QDRANT_DISTANCE", "cosine") or "cosine").lower(),
            vector_timeout=_env_int("QDRANT_TIMEOUT", 30),
            graph_store_url=_env("NEO4J_URI", "bolt://localhost:7687")
            or "bolt://localhost:7687",
            graph_username=_env("NEO4J_USERNAME", "neo4j") or "neo4j",
            graph_password=_env("NEO4J_PASSWORD", "password") or "password",
            graph_database=_env("NEO4J_DATABASE", "neo4j") or "neo4j",
            graph_connection_timeout=_env_int("NEO4J_CONNECTION_TIMEOUT", 60),
            graph_max_connection_lifetime=_env_int("NEO4J_MAX_CONNECTION_LIFETIME", 3600),
            graph_max_connection_pool_size=_env_int("NEO4J_MAX_CONNECTION_POOL_SIZE", 50),
            document_store_path=_env("DOCUMENT_STORE_PATH", "data/memories.db")
            or "data/memories.db",
            working_memory_ttl=_env_int("WORKING_MEMORY_TTL", 300),
            max_working_memory_items=_env_int("MAX_WORKING_MEMORY_ITEMS", 100),
        )


class BaseMemory(ABC):
    """记忆后端的抽象基类。
    
    定义所有记忆存储后端必须实现的接口。
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """初始化记忆后端。
        
        参数:
            config: 记忆配置，默认为从环境变量读取的配置
        """
        self.config = config or MemoryConfig.from_env()

    @abstractmethod
    def store(self, item: MemoryItem) -> str:
        """存储单个记忆项。
        
        参数:
            item: 要存储的记忆项
            
        返回:
            存储的记忆项ID
        """

    @abstractmethod
    def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """检索相关的记忆项。
        
        参数:
            query: 查询文本
            limit: 返回结果的最大数量
            
        返回:
            相关的记忆项列表
        """

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """删除单个记忆项。
        
        参数:
            memory_id: 要删除的记忆项ID
            
        返回:
            是否成功删除
        """

    @abstractmethod
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """更新单个记忆项。
        
        参数:
            memory_id: 要更新的记忆项ID
            updates: 更新字段的字典
            
        返回:
            是否成功更新
        """

    @abstractmethod
    def clear(self) -> bool:
        """清除所有记忆项。
        
        返回:
            是否成功清除
        """

    @abstractmethod
    def get_all_items(self) -> List[MemoryItem]:
        """获取所有记忆项。
        
        返回:
            所有记忆项的列表
        """

    @abstractmethod
    def search_by_keyword(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """关键词搜索。
        
        参数:
            query: 查询关键词
            limit: 返回结果的最大数量
            
        返回:
            匹配的记忆项列表
        """

    @abstractmethod
    def search_by_semantic(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """语义搜索。
        
        参数:
            query: 查询文本
            limit: 返回结果的最大数量
            
        返回:
            语义相关的记忆项列表
        """

    def search_hybrid(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """混合搜索：结合关键词和语义搜索结果并去重。
        
        参数:
            query: 查询文本
            limit: 返回结果的最大数量
            
        返回:
            去重后的混合搜索结果列表
        """
        keyword_results = self.search_by_keyword(query, limit=limit)
        semantic_results = self.search_by_semantic(query, limit=limit)

        all_results: Dict[str, MemoryItem] = {}
        for item in keyword_results + semantic_results:
            all_results[item.id] = item

        return list(all_results.values())[:limit]

    def score_item(self, query: str, item: MemoryItem) -> Optional[float]:
        """为单条记忆计算与查询相关的分数。

        说明:
            - 返回 ``None`` 表示该记忆后端不提供自己的排序分数，调用方可以使用外部默认排序逻辑。
            - 工作记忆会覆盖此方法，以便使用其内部的时间衰减和重要性公式。

        参数:
            query: 查询文本
            item: 待评分记忆项

        返回:
            分数或 ``None``
        """
        del query, item
        return None
