"""Plan-and-solve agent skeleton."""

from __future__ import annotations

from src.core.agent import Agent
from src.core.message import Message


class PlanAndSolveAgent(Agent):
    """Agent that asks the model to plan before answering."""

    def run(self, user_input: str) -> Message:
        self.ensure_system_prompt()
        self.add_message(Message(role="user", content=user_input))

        plan_request = Message(
            role="user",
            content=(
                "Create a concise plan only for the user's question. "
                "Do not solve it yet."
            ),
        )
        self.add_message(plan_request)
        plan = self.llm.generate(self._history)
        self.add_message(plan)

        solve_request = Message(
            role="user",
            content=(
                "Using the user's question and the plan above, provide the final answer."
            ),
        )
        self.add_message(solve_request)
        final_response = self.llm.generate(self._history)
        self.add_message(final_response)
        return final_response
