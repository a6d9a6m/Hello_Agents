"""Simple calculator tool."""

from __future__ import annotations

from src.tools.base import Tool, ToolParameter, ToolResult


class CalculatorTool(Tool):
    """Evaluate a basic Python arithmetic expression."""

    name = "calculator"
    description = "Evaluate arithmetic expressions safely."
    parameters = [
        ToolParameter(
            name="expression",
            param_type=str,
            description="Arithmetic expression to evaluate.",
        )
    ]

    def execute(self, **validated) -> ToolResult:
        expression = validated["expression"]
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
