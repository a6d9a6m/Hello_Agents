"""Plan-and-solve agent skeleton."""

from __future__ import annotations

from core.agent import Agent
from core.message import Message


class PlanAndSolveAgent(Agent):
    """Agent that asks the model to plan before answering."""

    def run(self, user_input: str) -> Message:
        planning_prompt = Message(
            role="user",
            content=(
                "Create a concise plan first, then solve the task.\n"
                f"Task: {user_input}"
            ),
        )
        self.add_message(planning_prompt)
        response = self.llm.generate(self.memory)
        self.add_message(response)
        return response
