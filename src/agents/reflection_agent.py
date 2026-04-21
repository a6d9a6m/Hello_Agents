"""反思Agent骨架"""

from __future__ import annotations

from src.core.agent import Agent
from src.core.message import Message


class ReflectionAgent(Agent):
    """执行初稿和反思两轮处理的Agent"""

    def run(self, user_input: str) -> Message:
        self.ensure_system_prompt()
        self.add_message(Message(role="user", content=user_input))

        draft_request = Message(
            role="user",
            content="为用户的问题撰写初稿答案。",
        )
        self.add_message(draft_request)
        draft = self.llm.generate(self._history)
        self.add_message(draft)

        reflection_prompt = Message(
            role="user",
            content=(
                "对初稿的正确性、缺失信息和清晰度进行反思。"
                "暂时不要重写。"
            ),
        )
        self.add_message(reflection_prompt)
        reflection = self.llm.generate(self._history)
        self.add_message(reflection)

        revision_prompt = Message(
            role="user",
            content="基于上面的反思修订初稿，提供最终答案。",
        )
        self.add_message(revision_prompt)
        final_response = self.llm.generate(self._history)
        self.add_message(final_response)
        return final_response
