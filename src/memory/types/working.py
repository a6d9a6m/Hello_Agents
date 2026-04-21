"""工作记忆实现"""

from __future__ import annotations

import time
from typing import List, Dict, Any, Optional
from collections import OrderedDict

from src.memory.base import BaseMemory, MemoryItem, MemoryType, MemoryConfig


class WorkingMemory(BaseMemory):
    """工作记忆：短期记忆，具有TTL管理"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        super().__init__(config)
        self.memories: OrderedDict[str, MemoryItem] = OrderedDict()
        self.ttl = config.working_memory_ttl if config else 300
        self.max_items = config.max_working_memory_items if config else 100
    
    def store(self, item: MemoryItem) -> str:
        """存储工作记忆项"""
        # 清理过期项
        self._cleanup()
        
        # 检查是否达到最大容量
        if len(self.memories) >= self.max_items:
            # 移除最旧的项
            oldest_key = next(iter(self.memories))
            del self.memories[oldest_key]
        
        # 存储新项
        self.memories[item.id] = item
        return item.id
    
    def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """检索工作记忆项（简单关键词匹配）"""
        # 清理过期项
        self._cleanup()
        
        # 简单关键词匹配
        results = []
        query_lower = query.lower()
        
        for item in reversed(self.memories.values()):  # 从最新到最旧
            if query_lower in item.content.lower():
                results.append(item)
                if len(results) >= limit:
                    break
        
        return results
    
    def delete(self, memory_id: str) -> bool:
        """删除工作记忆项"""
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """更新工作记忆项"""
        if memory_id in self.memories:
            item = self.memories[memory_id]
            
            # 更新字段
            for key, value in updates.items():
                if hasattr(item, key):
                    setattr(item, key, value)
            
            # 更新更新时间
            from datetime import datetime
            item.updated_at = datetime.now()
            
            # 移动到最新位置（LRU策略）
            self.memories.move_to_end(memory_id)
            
            return True
        return False
    
    def clear(self) -> bool:
        """清空工作记忆"""
        self.memories.clear()
        return True
    
    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        """获取最近的工作记忆项"""
        self._cleanup()
        
        items = list(self.memories.values())
        return list(reversed(items))[:limit]  # 返回最新的项
    
    def _cleanup(self):
        """清理过期的工作记忆项"""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self.memories.items():
            if item.ttl is not None:
                item_age = current_time - item.created_at.timestamp()
                if item_age > item.ttl:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self.memories[key]
    
    def get_all_items(self) -> List[MemoryItem]:
        """获取所有工作记忆项"""
        self._cleanup()
        return list(self.memories.values())
    
    def search_by_keyword(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """关键词搜索"""
        self._cleanup()
        
        results = []
        query_lower = query.lower()
        
        for item in reversed(self.memories.values()):  # 从最新到最旧
            # 检查内容中的关键词
            if query_lower in item.content.lower():
                results.append(item)
            # 检查标签中的关键词
            elif any(query_lower in tag.lower() for tag in item.tags):
                results.append(item)
            
            if len(results) >= limit:
                break
        
        return results
    
    def search_by_semantic(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """语义搜索（工作记忆使用简单相似度）"""
        self._cleanup()
        
        # 对于工作记忆，使用简单的文本相似度
        results = []
        query_lower = query.lower()
        
        for item in reversed(self.memories.values()):
            # 计算简单的文本重叠度
            content_lower = item.content.lower()
            words_query = set(query_lower.split())
            words_content = set(content_lower.split())
            
            if words_query and words_content:
                overlap = len(words_query.intersection(words_content)) / len(words_query)
                if overlap > 0.3:  # 30%重叠阈值
                    results.append(item)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        self._cleanup()
        return {
            "total_items": len(self.memories),
            "max_capacity": self.max_items,
            "ttl_seconds": self.ttl
        }