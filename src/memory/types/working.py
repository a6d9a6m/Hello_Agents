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
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        self._cleanup()
        return {
            "total_items": len(self.memories),
            "max_capacity": self.max_items,
            "ttl_seconds": self.ttl
        }