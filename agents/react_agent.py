"""ReAct-style agent skeleton."""

from __future__ import annotations

from core.agent import Agent
from core.message import Message
from tools.registry import ToolRegistry


class ReActAgent(Agent):
    """A lightweight ReAct-style agent with tool hints."""

    def __init__(self, llm, tool_registry: ToolRegistry, name: str | None = None) -> None:
        super().__init__(llm=llm, name=name)
        self.tool_registry = tool_registry

    def run(self, user_input: str) -> Message:
        tool_names = ", ".join(self.tool_registry.list_tools()) or "no tools"
        prompt = (
            "You are a ReAct-style agent.\n"
            f"Available tools: {tool_names}\n"
            f"User question: {user_input}"
        )
        self.add_message(Message(role="user", content=prompt))
        response = self.llm.generate(self.memory)
        self.add_message(response)
        return response
