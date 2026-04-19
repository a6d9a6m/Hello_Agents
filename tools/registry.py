"""Tool registration and lookup."""

from __future__ import annotations

from collections.abc import Iterable

from core.exceptions import ToolError
from tools.base import BaseTool


class ToolRegistry:
    """Registry that stores available tools by name."""

    def __init__(self, tools: Iterable[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> BaseTool:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise ToolError(f"Tool '{tool_name}' is not registered.") from exc

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())
