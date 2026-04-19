"""Async wrapper for running multiple tools concurrently."""

from __future__ import annotations

import asyncio
from typing import Any

from tools.base import ToolResult
from tools.registry import ToolRegistry


class AsyncToolExecutor:
    """Execute registered tools concurrently via asyncio."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def run_many(
        self,
        tasks: list[tuple[str, dict[str, Any]]],
    ) -> list[ToolResult]:
        coroutines = [self._run_single(tool_name, params) for tool_name, params in tasks]
        return await asyncio.gather(*coroutines)

    async def _run_single(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        tool = self.registry.get(tool_name)
        return await asyncio.to_thread(tool.run, **params)
