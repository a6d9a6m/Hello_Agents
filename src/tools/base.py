"""Tool base abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.core.exceptions import ToolError


@dataclass(slots=True)
class ToolResult:
    """Normalized tool execution result."""

    tool_name: str
    output: Any
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


_UNSET = object()


@dataclass(slots=True)
class ToolParameter:
    """Declarative tool parameter definition."""

    name: str
    param_type: type
    description: str = ""
    required: bool = True
    default: Any = _UNSET
    enum: list[Any] | None = None

    def validate(self, value: Any) -> Any:
        """Validate a single parameter value."""
        expected_type = self.param_type
        if expected_type not in {str, int, float, bool, list, dict}:
            raise ToolError(f"Unsupported parameter type for '{self.name}'.")

        if expected_type is bool:
            if type(value) is not bool:
                raise ToolError(f"Parameter '{self.name}' must be of type bool.")
        elif expected_type is int:
            if type(value) is not int:
                raise ToolError(f"Parameter '{self.name}' must be of type int.")
        elif not isinstance(value, expected_type):
            raise ToolError(
                f"Parameter '{self.name}' must be of type {expected_type.__name__}."
            )

        if self.enum is not None and value not in self.enum:
            raise ToolError(
                f"Parameter '{self.name}' must be one of: {', '.join(map(str, self.enum))}."
            )

        return value

    def has_default(self) -> bool:
        return self.default is not _UNSET

    def to_json_schema(self) -> dict[str, Any]:
        """Convert the parameter to a JSON schema fragment."""
        type_names = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        if self.param_type not in type_names:
            raise ToolError(f"Unsupported parameter type for '{self.name}'.")

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
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool interface"
    parameters: list[ToolParameter] = []

    def run(self, **kwargs: Any) -> ToolResult:
        """Validate inputs then execute the tool."""
        validated = self._validate_kwargs(kwargs)
        return self.execute(**validated)

    def _validate_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        parameter_map = {parameter.name: parameter for parameter in self.parameters}
        unexpected = sorted(set(kwargs) - set(parameter_map))
        if unexpected:
            raise ToolError(
                f"Unexpected parameters for tool '{self.name}': {', '.join(unexpected)}."
            )

        validated: dict[str, Any] = {}
        for parameter in self.parameters:
            if parameter.name in kwargs:
                validated[parameter.name] = parameter.validate(kwargs[parameter.name])
                continue

            if parameter.required:
                raise ToolError(
                    f"Missing required parameter '{parameter.name}' for tool '{self.name}'."
                )

            if parameter.has_default():
                validated[parameter.name] = parameter.default

        return validated

    def get_definition(self) -> dict[str, Any]:
        """Return the internal tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [parameter.to_json_schema() | {"name": parameter.name} for parameter in self.parameters],
        }

    def to_openai_schema(self) -> dict[str, Any]:
        """Return an OpenAI function tool schema."""
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
        """Execute the tool with validated keyword arguments."""


BaseTool = Tool
