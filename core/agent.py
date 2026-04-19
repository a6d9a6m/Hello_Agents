"""Base agent abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.llm import HelloAgentsLLM
from core.message import Message


class Agent(ABC):
    """Base class for all agent implementations."""

    def __init__(self, llm: HelloAgentsLLM, name: str | None = None) -> None:
        self.llm = llm
        self.name = name or self.__class__.__name__
        self.memory: list[Message] = []

    def reset(self) -> None:
        self.memory.clear()

    def add_message(self, message: Message) -> None:
        self.memory.append(message)

    @abstractmethod
    def run(self, user_input: str) -> Message:
        """Execute the agent for a single user input."""
