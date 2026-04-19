"""Unified LLM interface used by agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from core.message import Message


class HelloAgentsLLM(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def generate(self, messages: Iterable[Message], **kwargs: object) -> Message:
        """Return the assistant response for the given message list."""


class MockLLM(HelloAgentsLLM):
    """Minimal local LLM stub for project bootstrapping and tests."""

    def generate(self, messages: Iterable[Message], **kwargs: object) -> Message:
        last_message = ""
        for message in messages:
            if message.role == "user":
                last_message = message.content
        return Message(role="assistant", content=f"[mock-response] {last_message}")
