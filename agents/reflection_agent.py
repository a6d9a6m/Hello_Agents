"""Reflection agent skeleton."""

from __future__ import annotations

from core.agent import Agent
from core.message import Message


class ReflectionAgent(Agent):
    """Agent that performs a first pass and a reflection pass."""

    def run(self, user_input: str) -> Message:
        self.add_message(Message(role="user", content=user_input))
        draft = self.llm.generate(self.memory)
        self.add_message(draft)

        reflection_prompt = Message(
            role="user",
            content="Reflect on the previous answer and improve correctness and clarity.",
        )
        self.add_message(reflection_prompt)
        final_response = self.llm.generate(self.memory)
        self.add_message(final_response)
        return final_response
