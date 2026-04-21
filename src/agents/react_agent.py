"""ReAct风格Agent骨架"""

from __future__ import annotations

import re

from src.core.agent import Agent
from src.core.exceptions import ToolError
from src.core.message import Message
from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry


class ReActAgent(Agent):
    """带有工具提示的轻量级ReAct风格Agent"""

    def __init__(
        self,
        name: str,
        llm,
        tool_registry: ToolRegistry,
        system_prompt: str | None = None,
        config=None,
    ) -> None:
        super().__init__(name=name, llm=llm, system_prompt=system_prompt, config=config)
        self.tool_registry = tool_registry

    def run(self, user_input: str) -> Message:
        self.ensure_system_prompt()
        self.add_message(Message(role="user", content=user_input))

        tool_names = self.tool_registry.list_tools()
        if not tool_names:
            return self._answer_without_tools()

        react_prompt = Message(
            role="user",
            content=(
                "请以ReAct格式思考，并严格按照以下字段响应：\n"
                "思考：<简要推理>\n"
                f"动作：<从{', '.join(tool_names)}中选择一个>\n"
                "动作输入：<工具输入>"
            ),
        )
        self.add_message(react_prompt)
        reaction = self.llm.generate(self._history)
        self.add_message(reaction)

        action, action_input = self._parse_action(reaction.content)
        if not action:
            return self._fallback_final_answer()

        try:
            tool_result = self.tool_registry.execute(action, **self._build_tool_kwargs(action, action_input))
        except ToolError:
            return self._fallback_final_answer()

        self.add_message(self._tool_message(tool_result))
        final_prompt = Message(
            role="user",
            content="基于上面的工具观察结果，为用户提供最终答案。",
        )
        self.add_message(final_prompt)
        final_response = self.llm.generate(self._history)
        self.add_message(final_response)
        return final_response

    def _answer_without_tools(self) -> Message:
        direct_prompt = Message(
            role="user",
            content=(
                "没有可用工具。请在内部使用ReAct风格思考，但只返回最终答案。"
            ),
        )
        self.add_message(direct_prompt)
        response = self.llm.generate(self._history)
        self.add_message(response)
        return response

    def _fallback_final_answer(self) -> Message:
        fallback_prompt = Message(
            role="user",
            content="根据当前的对话内容，直接提供最佳最终答案。",
        )
        self.add_message(fallback_prompt)
        response = self.llm.generate(self._history)
        self.add_message(response)
        return response

    def _parse_action(self, content: str) -> tuple[str | None, str]:
        # 支持中文和英文格式，使用更宽松的匹配
        lines = content.split('\n')
        action = None
        action_input = ""
        
        for line in lines:
            line = line.strip()
            # 匹配动作行（需要精确匹配行开头）
            if line.startswith('动作:') or line.startswith('动作：') or line.startswith('Action:'):
                # 找到分隔符位置
                if line.startswith('动作:'):
                    action = line[3:].strip()  # 跳过"动作:"
                elif line.startswith('动作：'):
                    action = line[3:].strip()  # 跳过"动作："（注意中文字符长度）
                elif line.startswith('Action:'):
                    action = line[7:].strip()  # 跳过"Action:"
            # 匹配动作输入行
            elif line.startswith('动作输入:') or line.startswith('动作输入：') or line.startswith('Action Input:'):
                if line.startswith('动作输入:'):
                    action_input = line[5:].strip()  # 跳过"动作输入:"
                elif line.startswith('动作输入：'):
                    action_input = line[5:].strip()  # 跳过"动作输入："（注意中文字符长度）
                elif line.startswith('Action Input:'):
                    action_input = line[13:].strip()  # 跳过"Action Input:"
        
        return action, action_input

    def _build_tool_kwargs(self, action: str, action_input: str) -> dict[str, str]:
        tool = self.tool_registry.get(action)
        if not tool.parameters:
            return {}
        if len(tool.parameters) == 1:
            return {tool.parameters[0].name: action_input}
        if any(parameter.name == "input" for parameter in tool.parameters):
            return {"input": action_input}
        raise ToolError(f"工具 '{action}' 需要显式的参数映射。")

    def _tool_message(self, result: object) -> Message:
        if isinstance(result, ToolResult):
            content = str(result.output)
            metadata = {
                "tool_name": result.tool_name,
                "success": result.success,
            }
            if result.metadata:
                metadata["details"] = result.metadata
            return Message(role="tool", content=content, metadata=metadata)
        return Message(role="tool", content=str(result), metadata={"tool_name": "unknown", "success": True})
