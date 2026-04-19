"""Simple agent implementation."""

from __future__ import annotations

from core.agent import Agent
from core.message import Message


class SimpleAgent(Agent):
    """A minimal single-pass agent."""

    def run(self, user_input: str) -> Message:
        self.add_message(Message(role="user", content=user_input))
        response = self.llm.generate(self.memory)
        self.add_message(response)
        return response
