"""工具基础抽象"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.core.exceptions import ToolError


@dataclass(slots=True)
class ToolResult:
    """标准化的工具执行结果"""

    tool_name: str
    output: Any
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


_UNSET = object()


@dataclass(slots=True)
class ToolParameter:
    """声明式工具参数定义"""

    name: str
    param_type: type
    description: str = ""
    required: bool = True
    default: Any = _UNSET
    enum: list[Any] | None = None

    def validate(self, value: Any) -> Any:
        """验证单个参数值"""
        expected_type = self.param_type
        if expected_type not in {str, int, float, bool, list, dict}:
            raise ToolError(f"参数 '{self.name}' 的类型不支持。")

        if expected_type is bool:
            if type(value) is not bool:
                raise ToolError(f"参数 '{self.name}' 必须是布尔类型。")
        elif expected_type is int:
            if type(value) is not int:
                raise ToolError(f"参数 '{self.name}' 必须是整数类型。")
        elif not isinstance(value, expected_type):
            raise ToolError(
                f"参数 '{self.name}' 必须是 {expected_type.__name__} 类型。"
            )

        if self.enum is not None and value not in self.enum:
            raise ToolError(
                f"参数 '{self.name}' 必须是以下值之一：{', '.join(map(str, self.enum))}。"
            )

        return value

    def has_default(self) -> bool:
        return self.default is not _UNSET

    def to_json_schema(self) -> dict[str, Any]:
        """将参数转换为JSON Schema片段"""
        type_names = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        if self.param_type not in type_names:
            raise ToolError(f"参数 '{self.name}' 的类型不支持。")

        schema: dict[str, Any] = {
            "type": type_names[self.param_type],
            "description": self.description,
        }
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if not self.required and self.has_default():
            schema["default"] = self.default
        return schema


class Tool(ABC):
    """所有工具的基础类"""

    name: str = "base_tool"
    description: str = "基础工具接口"
    parameters: list[ToolParameter] = []

    def run(self, **kwargs: Any) -> ToolResult:
        """验证输入然后执行工具"""
        validated = self._validate_kwargs(kwargs)
        return self.execute(**validated)

    def _validate_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        parameter_map = {parameter.name: parameter for parameter in self.parameters}
        unexpected = sorted(set(kwargs) - set(parameter_map))
        if unexpected:
            raise ToolError(
                f"工具 '{self.name}' 接收到意外参数：{', '.join(unexpected)}。"
            )

        validated: dict[str, Any] = {}
        for parameter in self.parameters:
            if parameter.name in kwargs:
                validated[parameter.name] = parameter.validate(kwargs[parameter.name])
                continue

            if parameter.required:
                raise ToolError(
                    f"工具 '{self.name}' 缺少必需参数 '{parameter.name}'。"
                )

            if parameter.has_default():
                validated[parameter.name] = parameter.default

        return validated

    def get_definition(self) -> dict[str, Any]:
        """返回内部工具定义"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [parameter.to_json_schema() | {"name": parameter.name} for parameter in self.parameters],
        }

    def to_openai_schema(self) -> dict[str, Any]:
        """返回OpenAI函数工具schema"""
        properties = {
            parameter.name: parameter.to_json_schema() for parameter in self.parameters
        }
        required = [parameter.name for parameter in self.parameters if parameter.required]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @abstractmethod
    def execute(self, **validated: Any) -> ToolResult:
        """使用已验证的关键字参数执行工具"""


BaseTool = Tool
