"""简单的本地搜索占位符"""

from __future__ import annotations

from src.tools.base import Tool, ToolParameter, ToolResult


class SearchTool(Tool):
    """用于本地开发的存根搜索工具"""

    name = "search"
    description = "为早期开发返回模拟搜索结果。"
    parameters = [
        ToolParameter(
            name="query",
            param_type=str,
            description="搜索查询字符串。",
        )
    ]

    def execute(self, **validated) -> ToolResult:
        query = validated["query"]
        if not query:
            return ToolResult(tool_name=self.name, output="查询为空。", success=False)
        return ToolResult(
            tool_name=self.name,
            output=f"查询 '{query}' 的模拟搜索结果",
        )
