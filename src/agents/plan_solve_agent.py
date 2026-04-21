"""计划-求解Agent骨架"""

from __future__ import annotations

from src.core.agent import Agent
from src.core.message import Message


class PlanAndSolveAgent(Agent):
    """要求模型先制定计划再回答的Agent"""

    def run(self, user_input: str) -> Message:
        self.ensure_system_prompt()
        self.add_message(Message(role="user", content=user_input))

        plan_request = Message(
            role="user",
            content=(
                "请仅为用户的问题制定一个简洁的计划。"
                "暂时不要解决问题。"
            ),
        )
        self.add_message(plan_request)
        plan = self.llm.generate(self._history)
        self.add_message(plan)

        solve_request = Message(
            role="user",
            content=(
                "基于用户的问题和上面的计划，提供最终答案。"
            ),
        )
        self.add_message(solve_request)
        final_response = self.llm.generate(self._history)
        self.add_message(final_response)
        return final_response
