"""简单计算器工具"""

from __future__ import annotations

from src.tools.base import Tool, ToolParameter, ToolResult


class CalculatorTool(Tool):
    """评估基本的Python算术表达式"""

    name = "calculator"
    description = "安全地评估算术表达式。"
    parameters = [
        ToolParameter(
            name="expression",
            param_type=str,
            description="要评估的算术表达式。",
        )
    ]

    def execute(self, **validated) -> ToolResult:
        expression = validated["expression"]
        allowed_chars = set("0123456789+-*/(). %")
        if not expression or any(char not in allowed_chars for char in expression):
            return ToolResult(
                tool_name=self.name,
                output="不支持的表达式。",
                success=False,
            )
        try:
            result = eval(expression, {"__builtins__": {}}, {})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(tool_name=self.name, output=str(exc), success=False)
        return ToolResult(tool_name=self.name, output=result)
