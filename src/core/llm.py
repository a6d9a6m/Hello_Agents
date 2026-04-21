"""LLM abstractions for Hello_Agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Sequence

from .message import Message


class HelloAgentsLLM(ABC):
    """Base LLM interface for Hello_Agents."""

    @abstractmethod
    def generate(self, messages: Sequence[Message]) -> Message:
        """Generate a response given a conversation history."""


class MockLLM(HelloAgentsLLM):
    """Mock LLM that echoes the last user message."""

    def generate(self, messages: Sequence[Message]) -> Message:
        if not messages:
            return Message(role="assistant", content="No messages provided.")
        
        last_message = messages[-1]
        if last_message.role == "user":
            return Message(
                role="assistant",
                content=f"Mock response to: {last_message.content}"
            )
        
        return Message(
            role="assistant",
            content="I received a non-user message."
        )