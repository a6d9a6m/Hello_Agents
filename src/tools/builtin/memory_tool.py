"""记忆工具（Agent记忆能力）"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from enum import Enum

from src.tools.base import Tool, ToolParameter, ToolResult
from src.memory.manager import MemoryManager
from src.memory.base import MemoryType


class MemoryOperation(str, Enum):
    """记忆操作类型"""
    STORE = "store"
    RETRIEVE = "retrieve"
    DELETE = "delete"
    CLEAR = "clear"
    STATS = "stats"


class MemoryTool(Tool):
    """记忆工具：为Agent提供记忆能力"""
    
    name = "memory"
    description = "管理Agent的记忆系统，支持存储、检索、删除记忆"
    
    parameters = [
        ToolParameter(
            name="operation",
            param_type=str,
            description="记忆操作类型：store(存储)、retrieve(检索)、delete(删除)、clear(清空)、stats(统计)",
            required=True,
            enum=[op.value for op in MemoryOperation]
        ),
        ToolParameter(
            name="content",
            param_type=str,
            description="要存储或检索的内容",
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
            name="memory_id",
            param_type=str,
            description="记忆ID（用于删除操作）",
            required=False
        ),
        ToolParameter(
            name="limit",
            param_type=int,
            description="检索结果数量限制",
            required=False,
            default=5
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
    
    def execute(self, **validated) -> ToolResult:
        """执行记忆操作"""
        operation = validated["operation"]
        
        try:
            if operation == MemoryOperation.STORE.value:
                result = self._store_memory(validated)
            elif operation == MemoryOperation.RETRIEVE.value:
                result = self._retrieve_memory(validated)
            elif operation == MemoryOperation.DELETE.value:
                result = self._delete_memory(validated)
            elif operation == MemoryOperation.CLEAR.value:
                result = self._clear_memory(validated)
            elif operation == MemoryOperation.STATS.value:
                result = self._get_stats(validated)
            else:
                result = ToolResult(
                    tool_name=self.name,
                    output=f"不支持的操作：{operation}",
                    success=False
                )
            
            return result
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                output=f"记忆操作失败：{str(e)}",
                success=False
            )
    
    def _store_memory(self, params: Dict[str, Any]) -> ToolResult:
        """存储记忆"""
        content = params.get("content")
        if not content:
            return ToolResult(
                tool_name=self.name,
                output="存储记忆需要提供content参数",
                success=False
            )
        
        memory_type = MemoryType(params.get("memory_type", "working"))
        metadata = params.get("metadata", {})
        
        memory_id = self.memory_manager.store(
            content=content,
            memory_type=memory_type,
            **metadata
        )
        
        return ToolResult(
            tool_name=self.name,
            output=f"记忆存储成功，ID：{memory_id}",
            success=True,
            metadata={
                "memory_id": memory_id,
                "memory_type": memory_type.value,
                "content_length": len(content)
            }
        )
    
    def _retrieve_memory(self, params: Dict[str, Any]) -> ToolResult:
        """检索记忆"""
        query = params.get("content", "")
        memory_type_str = params.get("memory_type")
        limit = params.get("limit", 5)
        
        memory_types = None
        if memory_type_str:
            memory_types = [MemoryType(memory_type_str)]
        
        memories = self.memory_manager.retrieve(
            query=query,
            memory_types=memory_types,
            limit=limit
        )
        
        if not memories:
            return ToolResult(
                tool_name=self.name,
                output="未找到相关记忆",
                success=True
            )
        
        # 格式化输出
        output_parts = [f"找到 {len(memories)} 条相关记忆："]
        for i, memory in enumerate(memories, 1):
            output_parts.append(f"\n{i}. [{memory.memory_type.value}] {memory.content[:100]}...")
            if memory.metadata:
                output_parts.append(f"   元数据：{memory.metadata}")
        
        return ToolResult(
            tool_name=self.name,
            output="\n".join(output_parts),
            success=True,
            metadata={
                "count": len(memories),
                "memory_types": [m.memory_type.value for m in memories]
            }
        )
    
    def _delete_memory(self, params: Dict[str, Any]) -> ToolResult:
        """删除记忆"""
        memory_id = params.get("memory_id")
        memory_type_str = params.get("memory_type", "working")
        
        if not memory_id:
            return ToolResult(
                tool_name=self.name,
                output="删除记忆需要提供memory_id参数",
                success=False
            )
        
        memory_type = MemoryType(memory_type_str)
        success = self.memory_manager.delete(memory_id, memory_type)
        
        if success:
            return ToolResult(
                tool_name=self.name,
                output=f"记忆删除成功，ID：{memory_id}",
                success=True
            )
        else:
            return ToolResult(
                tool_name=self.name,
                output=f"记忆删除失败，ID：{memory_id}",
                success=False
            )
    
    def _clear_memory(self, params: Dict[str, Any]) -> ToolResult:
        """清空记忆"""
        memory_type_str = params.get("memory_type")
        
        if memory_type_str:
            memory_type = MemoryType(memory_type_str)
            success = self.memory_manager.clear(memory_type)
            operation = f"{memory_type.value}记忆"
        else:
            success = self.memory_manager.clear()
            operation = "所有记忆"
        
        if success:
            return ToolResult(
                tool_name=self.name,
                output=f"{operation}清空成功",
                success=True
            )
        else:
            return ToolResult(
                tool_name=self.name,
                output=f"{operation}清空失败",
                success=False
            )
    
    def _get_stats(self, params: Dict[str, Any]) -> ToolResult:
        """获取统计信息"""
        stats = self.memory_manager.get_stats()
        
        output_parts = ["记忆系统统计："]
        for memory_type, stat in stats.items():
            output_parts.append(f"\n{memory_type}:")
            for key, value in stat.items():
                output_parts.append(f"  {key}: {value}")
        
        return ToolResult(
            tool_name=self.name,
            output="\n".join(output_parts),
            success=True,
            metadata=stats
        )
    
    def get_context(self, query: str, context_size: int = 3) -> str:
        """获取上下文信息（用于Agent）"""
        return self.memory_manager.retrieve_context(query, context_size)