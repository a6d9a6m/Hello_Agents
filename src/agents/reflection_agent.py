"""Reflection agent skeleton."""

from __future__ import annotations

from src.core.agent import Agent
from src.core.message import Message


class ReflectionAgent(Agent):
    """Agent that performs a first pass and a reflection pass."""

    def run(self, user_input: str) -> Message:
        self.ensure_system_prompt()
        self.add_message(Message(role="user", content=user_input))

        draft_request = Message(
            role="user",
            content="Write a first draft answer to the user's question.",
        )
        self.add_message(draft_request)
        draft = self.llm.generate(self._history)
        self.add_message(draft)

        reflection_prompt = Message(
            role="user",
            content=(
                "Reflect on the draft for correctness, missing information, and clarity. "
                "Do not rewrite it yet."
            ),
        )
        self.add_message(reflection_prompt)
        reflection = self.llm.generate(self._history)
        self.add_message(reflection)

        revision_prompt = Message(
            role="user",
            content="Revise the draft using the reflection above and provide the final answer.",
        )
        self.add_message(revision_prompt)
        final_response = self.llm.generate(self._history)
        self.add_message(final_response)
        return final_response
