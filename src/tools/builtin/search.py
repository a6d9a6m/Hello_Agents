"""Simple local search placeholder."""

from __future__ import annotations

from src.tools.base import Tool, ToolParameter, ToolResult


class SearchTool(Tool):
    """Stub search tool used for local development."""

    name = "search"
    description = "Return a mock search result for early-stage development."
    parameters = [
        ToolParameter(
            name="query",
            param_type=str,
            description="Search query string.",
        )
    ]

    def execute(self, **validated) -> ToolResult:
        query = validated["query"]
        if not query:
            return ToolResult(tool_name=self.name, output="Query is empty.", success=False)
        return ToolResult(
            tool_name=self.name,
            output=f"Mock search result for query: {query}",
        )
