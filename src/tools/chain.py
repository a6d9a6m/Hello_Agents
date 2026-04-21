"""Simple tool chain orchestration."""

from __future__ import annotations

from typing import Any

from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry


class ToolChain:
    """Execute a sequence of tools with independent inputs."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def run(self, steps: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for tool_name, params in steps:
            tool = self.registry.get(tool_name)
            results.append(tool.run(**params))
        return results
