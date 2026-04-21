"""记忆工具（Agent记忆能力）"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Union
from enum import Enum

from src.tools.base import Tool, ToolParameter, ToolResult
from src.memory.manager import MemoryManager
from src.memory.base import MemoryType


class MemoryOperation(str, Enum):
    """记忆操作类型"""
    ADD = "add"
    SEARCH = "search"
    SUMMARY = "summary"
    STATS = "stats"
    UPDATE = "update"
    REMOVE = "remove"
    FORGET = "forget"
    CONSOLIDATE = "consolidate"
    CLEAR_ALL = "clear_all"


class MemoryTool(Tool):
    """记忆工具：为Agent提供记忆能力"""
    
    name = "memory"
    description = "管理Agent的记忆系统，支持存储、检索、删除、遗忘、整合等多种记忆操作"
    
    parameters = [
        ToolParameter(
            name="action",
            param_type=str,
            description="记忆操作类型：add(添加)、search(搜索)、summary(摘要)、stats(统计)、update(更新)、remove(删除)、forget(遗忘)、consolidate(整合)、clear_all(清空)",
            required=True,
            enum=[op.value for op in MemoryOperation]
        ),
        ToolParameter(
            name="content",
            param_type=str,
            description="记忆内容（用于add/search操作）",
            required=False
        ),
        ToolParameter(
            name="memory_type",
            param_type=str,
            description="记忆类型：working(工作记忆)、episodic(情景记忆)、semantic(语义记忆)、perceptual(感知记忆)",
            required=False,
            default="working",
            enum=[mt.value for mt in MemoryType]
        ),
        ToolParameter(
            name="memory_types",
            param_type=list,
            description="多个记忆类型列表（用于search操作）",
            required=False
        ),
        ToolParameter(
            name="memory_id",
            param_type=str,
            description="记忆ID（用于update/remove操作）",
            required=False
        ),
        ToolParameter(
            name="limit",
            param_type=int,
            description="搜索结果数量限制",
            required=False,
            default=10
        ),
        ToolParameter(
            name="importance",
            param_type=float,
            description="重要性评分（0-1，用于add操作）",
            required=False,
            default=0.5
        ),
        ToolParameter(
            name="emotion",
            param_type=str,
            description="情感标签（用于add操作）",
            required=False
        ),
        ToolParameter(
            name="tags",
            param_type=list,
            description="关联标签列表（用于add操作）",
            required=False
        ),
        ToolParameter(
            name="location",
            param_type=str,
            description="位置信息（用于add操作）",
            required=False
        ),
        ToolParameter(
            name="source",
            param_type=str,
            description="来源信息（用于add操作）",
            required=False
        ),
        ToolParameter(
            name="search_mode",
            param_type=str,
            description="搜索模式：keyword(关键词)、semantic(语义)、hybrid(混合)",
            required=False,
            default="hybrid"
        ),
        ToolParameter(
            name="min_importance",
            param_type=float,
            description="最小重要性阈值（用于search操作）",
            required=False,
            default=0.0
        ),
        ToolParameter(
            name="forget_strategy",
            param_type=str,
            description="遗忘策略：importance_based(基于重要性)、time_based(基于时间)、capacity_based(基于容量)、combined(组合)",
            required=False,
            default="importance_based"
        ),
        ToolParameter(
            name="importance_threshold",
            param_type=float,
            description="重要性阈值（用于forget/consolidate操作）",
            required=False,
            default=0.3
        ),
        ToolParameter(
            name="days_old",
            param_type=int,
            description="天数阈值（用于time_based遗忘策略）",
            required=False,
            default=30
        ),
        ToolParameter(
            name="max_items",
            param_type=int,
            description="最大记忆项数（用于capacity_based遗忘策略）",
            required=False,
            default=1000
        ),
        ToolParameter(
            name="updates",
            param_type=dict,
            description="更新内容（用于update操作）",
            required=False
        ),
        ToolParameter(
            name="metadata",
            param_type=dict,
            description="记忆元数据（JSON格式）",
            required=False
        )
    ]
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        super().__init__()
        self.memory_manager = memory_manager or MemoryManager()
    
    def execute(self, action: str, **kwargs) -> str:
        """执行记忆操作
        
        支持的操作：
        - add: 添加记忆（支持4种类型: working/episodic/semantic/perceptual）
        - search: 搜索记忆
        - summary: 获取记忆摘要
        - stats: 获取统计信息
        - update: 更新记忆
        - remove: 删除记忆
        - forget: 遗忘记忆（多种策略）
        - consolidate: 整合记忆（短期→长期）
        - clear_all: 清空所有记忆
        """
        try:
            if action == MemoryOperation.ADD.value:
                result = self._add_memory(kwargs)
            elif action == MemoryOperation.SEARCH.value:
                result = self._search_memory(kwargs)
            elif action == MemoryOperation.SUMMARY.value:
                result = self._get_summary(kwargs)
            elif action == MemoryOperation.STATS.value:
                result = self._get_stats(kwargs)
            elif action == MemoryOperation.UPDATE.value:
                result = self._update_memory(kwargs)
            elif action == MemoryOperation.REMOVE.value:
                result = self._remove_memory(kwargs)
            elif action == MemoryOperation.FORGET.value:
                result = self._forget_memory(kwargs)
            elif action == MemoryOperation.CONSOLIDATE.value:
                result = self._consolidate_memory(kwargs)
            elif action == MemoryOperation.CLEAR_ALL.value:
                result = self._clear_all_memory(kwargs)
            else:
                result = f"不支持的操作：{action}"
            
            return result
            
        except Exception as e:
            return f"记忆操作失败：{str(e)}"
    
    def _add_memory(self, params: Dict[str, Any]) -> str:
        """添加记忆"""
        content = params.get("content")
        if not content:
            return "添加记忆需要提供content参数"
        
        memory_type_str = params.get("memory_type", "working")
        memory_type = MemoryType(memory_type_str)
        
        # 构建元数据
        metadata = params.get("metadata", {})
        
        # 添加上下文信息
        if "importance" in params:
            metadata["importance"] = params["importance"]
        if "emotion" in params:
            metadata["emotion"] = params["emotion"]
        if "tags" in params:
            metadata["tags"] = params["tags"]
        if "location" in params:
            metadata["location"] = params["location"]
        if "source" in params:
            metadata["source"] = params["source"]
        
        memory_id = self.memory_manager.store(
            content=content,
            memory_type=memory_type,
            **metadata
        )
        
        return f"记忆添加成功，ID：{memory_id}\n类型：{memory_type.value}\n内容：{content[:50]}..."
    
    def _search_memory(self, params: Dict[str, Any]) -> str:
        """搜索记忆"""
        query = params.get("content", "")
        if not query:
            return "搜索记忆需要提供content参数"
        
        # 处理记忆类型参数
        memory_types = None
        if "memory_types" in params:
            memory_types = [MemoryType(mt) for mt in params["memory_types"]]
        elif "memory_type" in params:
            memory_types = [MemoryType(params["memory_type"])]
        
        search_mode = params.get("search_mode", "hybrid")
        limit = params.get("limit", 10)
        min_importance = params.get("min_importance", 0.0)
        
        memories = self.memory_manager.search(
            query=query,
            memory_types=memory_types,
            search_mode=search_mode,
            limit=limit,
            min_importance=min_importance
        )
        
        if not memories:
            return f"未找到与'{query}'相关的记忆"
        
        # 格式化输出
        output_parts = [f"找到 {len(memories)} 条相关记忆（搜索模式：{search_mode}）："]
        
        for i, memory in enumerate(memories, 1):
            importance_str = f"重要性：{memory.importance:.2f}"
            emotion_str = f"情感：{memory.emotion}" if memory.emotion else ""
            tags_str = f"标签：{', '.join(memory.tags)}" if memory.tags else ""
            
            output_parts.append(f"\n{i}. [{memory.memory_type.value}] {memory.content[:80]}...")
            output_parts.append(f"   {importance_str} {emotion_str} {tags_str}")
            if memory.location:
                output_parts.append(f"   位置：{memory.location}")
        
        return "\n".join(output_parts)
    
    def _get_summary(self, params: Dict[str, Any]) -> str:
        """获取记忆摘要"""
        memory_type_str = params.get("memory_type")
        memory_type = MemoryType(memory_type_str) if memory_type_str else None
        
        summary = self.memory_manager.get_summary(memory_type)
        
        if memory_type:
            # 单个记忆类型摘要
            output_parts = [f"{memory_type.value}记忆摘要："]
            output_parts.append(f"总数：{summary['total_count']}")
            output_parts.append(f"平均重要性：{summary['avg_importance']}")
            output_parts.append(f"最近5条记忆：")
            for i, item in enumerate(summary['recent_items'], 1):
                output_parts.append(f"  {i}. {item}")
        else:
            # 所有记忆类型摘要
            output_parts = ["所有记忆类型摘要："]
            for mt, data in summary.items():
                output_parts.append(f"\n{mt}:")
                output_parts.append(f"  总数：{data['total_count']}")
                output_parts.append(f"  平均重要性：{data['avg_importance']}")
        
        return "\n".join(output_parts)
    
    def _get_stats(self, params: Dict[str, Any]) -> str:
        """获取统计信息"""
        stats = self.memory_manager.get_stats()
        
        output_parts = ["记忆系统详细统计："]
        
        # 总体统计
        overall = stats.pop("overall", {})
        output_parts.append(f"\n总体统计：")
        output_parts.append(f"  总记忆数：{overall.get('total_count', 0)}")
        output_parts.append(f"  平均重要性：{overall.get('avg_importance', 0)}")
        output_parts.append(f"  记忆类型数：{overall.get('memory_types', 0)}")
        
        # 各类型统计
        for memory_type, stat in stats.items():
            output_parts.append(f"\n{memory_type}记忆：")
            output_parts.append(f"  数量：{stat['count']}")
            output_parts.append(f"  平均重要性：{stat['avg_importance']}")
            output_parts.append(f"  最近7天新增：{stat['recent_count']}")
            
            if stat['emotion_distribution']:
                output_parts.append(f"  情感分布：")
                for emotion, count in stat['emotion_distribution'].items():
                    output_parts.append(f"    {emotion}: {count}")
        
        return "\n".join(output_parts)
    
    def _update_memory(self, params: Dict[str, Any]) -> str:
        """更新记忆"""
        memory_id = params.get("memory_id")
        memory_type_str = params.get("memory_type", "working")
        updates = params.get("updates", {})
        
        if not memory_id:
            return "更新记忆需要提供memory_id参数"
        
        if not updates:
            return "更新记忆需要提供updates参数"
        
        memory_type = MemoryType(memory_type_str)
        success = self.memory_manager.update(memory_id, memory_type, updates)
        
        if success:
            return f"记忆更新成功，ID：{memory_id}\n更新内容：{updates}"
        else:
            return f"记忆更新失败，ID：{memory_id}"
    
    def _remove_memory(self, params: Dict[str, Any]) -> str:
        """删除记忆"""
        memory_id = params.get("memory_id")
        memory_type_str = params.get("memory_type", "working")
        
        if not memory_id:
            return "删除记忆需要提供memory_id参数"
        
        memory_type = MemoryType(memory_type_str)
        success = self.memory_manager.delete(memory_id, memory_type)
        
        if success:
            return f"记忆删除成功，ID：{memory_id}"
        else:
            return f"记忆删除失败，ID：{memory_id}"
    
    def _forget_memory(self, params: Dict[str, Any]) -> str:
        """遗忘记忆"""
        strategy = params.get("forget_strategy", "importance_based")
        
        # 准备策略参数
        strategy_params = {}
        if strategy in ["importance_based", "combined"]:
            strategy_params["importance_threshold"] = params.get("importance_threshold", 0.3)
        if strategy in ["time_based", "combined"]:
            strategy_params["days_old"] = params.get("days_old", 30)
        if strategy == "capacity_based":
            strategy_params["max_items"] = params.get("max_items", 1000)
        
        result = self.memory_manager.forget(strategy, **strategy_params)
        
        deleted_count = result["deleted_count"]
        strategy_used = result["strategy"]
        
        if deleted_count > 0:
            return f"遗忘操作完成\n策略：{strategy_used}\n删除记忆数：{deleted_count}"
        else:
            return f"没有需要遗忘的记忆\n策略：{strategy_used}"
    
    def _consolidate_memory(self, params: Dict[str, Any]) -> str:
        """整合记忆"""
        importance_threshold = params.get("importance_threshold", 0.7)
        
        result = self.memory_manager.consolidate(importance_threshold)
        
        consolidated_count = result["consolidated_count"]
        
        if consolidated_count > 0:
            return f"记忆整合完成\n重要性阈值：{importance_threshold}\n整合记忆数：{consolidated_count}"
        else:
            return f"没有需要整合的记忆\n重要性阈值：{importance_threshold}"
    
    def _clear_all_memory(self, params: Dict[str, Any]) -> str:
        """清空所有记忆"""
        success = self.memory_manager.clear()
        
        if success:
            return "所有记忆已清空"
        else:
            return "清空记忆失败"
    
    def get_context(self, query: str, context_size: int = 3) -> str:
        """获取上下文信息（用于Agent）"""
        return self.memory_manager.retrieve_context(query, context_size)