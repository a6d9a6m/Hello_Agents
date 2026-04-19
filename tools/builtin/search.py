"""Simple local search placeholder."""

from __future__ import annotations

from tools.base import BaseTool, ToolResult


class SearchTool(BaseTool):
    """Stub search tool used for local development."""

    name = "search"
    description = "Return a mock search result for early-stage development."

    def run(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(tool_name=self.name, output="Query is empty.", success=False)
        return ToolResult(
            tool_name=self.name,
            output=f"Mock search result for query: {query}",
        )
