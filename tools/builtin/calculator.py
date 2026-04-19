"""Simple calculator tool."""

from __future__ import annotations

from tools.base import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    """Evaluate a basic Python arithmetic expression."""

    name = "calculator"
    description = "Evaluate arithmetic expressions safely."

    def run(self, **kwargs) -> ToolResult:
        expression = kwargs.get("expression", "")
        allowed_chars = set("0123456789+-*/(). %")
        if not expression or any(char not in allowed_chars for char in expression):
            return ToolResult(
                tool_name=self.name,
                output="Unsupported expression.",
                success=False,
            )
        try:
            result = eval(expression, {"__builtins__": {}}, {})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(tool_name=self.name, output=str(exc), success=False)
        return ToolResult(tool_name=self.name, output=result)
