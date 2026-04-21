"""用于并发运行多个工具的异步包装器"""

from __future__ import annotations

import asyncio
from typing import Any

from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry


class AsyncToolExecutor:
    """通过asyncio并发执行已注册的工具"""

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
