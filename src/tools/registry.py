"""工具注册与查找"""

from __future__ import annotations

from collections.abc import Iterable
import inspect
from types import ModuleType

from src.core.exceptions import ToolError
from src.tools.base import Tool


class ToolRegistry:
    """按名称存储可用工具的注册表"""

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools:
            self.register_many(tools)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"工具 '{tool.name}' 已注册。")
        self._tools[tool.name] = tool

    def register_many(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, tool_name: str) -> Tool:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise ToolError(f"工具 '{tool_name}' 未注册。") from exc

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def execute(self, tool_name: str, **kwargs: object) -> object:
        return self.get(tool_name).run(**kwargs)

    def list_definitions(self) -> list[dict[str, object]]:
        return [self._tools[name].get_definition() for name in self.list_tools()]

    def list_openai_schemas(self) -> list[dict[str, object]]:
        return [self._tools[name].to_openai_schema() for name in self.list_tools()]

    def discover_from_module(self, module: ModuleType) -> list[Tool]:
        discovered: list[Tool] = []
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if not issubclass(candidate, Tool) or candidate is Tool:
                continue
            if candidate.__module__ != module.__name__:
                continue
            signature = inspect.signature(candidate)
            if any(parameter.default is inspect._empty for parameter in signature.parameters.values()):
                raise ToolError(
                    f"工具类 '{candidate.__name__}' 必须能够无参数实例化。"
                )
            tool = candidate()
            self.register(tool)
            discovered.append(tool)
        return discovered
