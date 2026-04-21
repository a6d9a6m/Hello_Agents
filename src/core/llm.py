"""Hello_Agents的LLM抽象层"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Sequence

from .message import Message


class HelloAgentsLLM(ABC):
    """Hello_Agents的基础LLM接口"""

    @abstractmethod
    def generate(self, messages: Sequence[Message]) -> Message:
        """根据对话历史生成响应"""


class MockLLM(HelloAgentsLLM):
    """模拟LLM，回显最后一条用户消息"""

    def generate(self, messages: Sequence[Message]) -> Message:
        if not messages:
            return Message(role="assistant", content="未提供消息。")
        
        last_message = messages[-1]
        if last_message.role == "user":
            return Message(
                role="assistant",
                content=f"模拟响应：{last_message.content}"
            )
        
        return Message(
            role="assistant",
            content="收到非用户消息。"
        )