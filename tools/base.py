"""Tool base abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolResult:
    """Normalized tool execution result."""

    tool_name: str
    output: Any
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool interface"

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with keyword arguments."""
