"""ReAct-style agent skeleton."""

from __future__ import annotations

import re

from src.core.agent import Agent
from src.core.exceptions import ToolError
from src.core.message import Message
from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry


class ReActAgent(Agent):
    """A lightweight ReAct-style agent with tool hints."""

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
                "Think in ReAct format and respond with exactly these fields:\n"
                "Thought: <brief reasoning>\n"
                f"Action: <one of {', '.join(tool_names)}>\n"
                "Action Input: <tool input>"
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
            content="Use the tool observation above to provide the final answer to the user.",
        )
        self.add_message(final_prompt)
        final_response = self.llm.generate(self._history)
        self.add_message(final_response)
        return final_response

    def _answer_without_tools(self) -> Message:
        direct_prompt = Message(
            role="user",
            content=(
                "No tools are available. Use a ReAct-style approach internally, but return only the final answer."
            ),
        )
        self.add_message(direct_prompt)
        response = self.llm.generate(self._history)
        self.add_message(response)
        return response

    def _fallback_final_answer(self) -> Message:
        fallback_prompt = Message(
            role="user",
            content="Provide the best final answer directly based on the conversation so far.",
        )
        self.add_message(fallback_prompt)
        response = self.llm.generate(self._history)
        self.add_message(response)
        return response

    def _parse_action(self, content: str) -> tuple[str | None, str]:
        action_match = re.search(r"^Action:\s*(.+)$", content, re.MULTILINE)
        input_match = re.search(
            r"^Action Input:\s*(.*?)(?:\n[A-Z][A-Za-z ]*:\s|\Z)",
            content,
            re.MULTILINE | re.DOTALL,
        )
        action = action_match.group(1).strip() if action_match else None
        action_input = input_match.group(1).strip() if input_match else ""
        return action, action_input

    def _build_tool_kwargs(self, action: str, action_input: str) -> dict[str, str]:
        tool = self.tool_registry.get(action)
        if not tool.parameters:
            return {}
        if len(tool.parameters) == 1:
            return {tool.parameters[0].name: action_input}
        if any(parameter.name == "input" for parameter in tool.parameters):
            return {"input": action_input}
        raise ToolError(f"Tool '{action}' requires explicit parameter mapping.")

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
