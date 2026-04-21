"""Agent基类"""
from abc import ABC, abstractmethod
from typing import Optional

from .config import Config
from .llm import HelloAgentsLLM
from .message import Message

class Agent(ABC):
    """Agent基类"""
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> Message:
        """运行Agent"""
        pass

    def ensure_system_prompt(self) -> None:
        """Add the configured system prompt once when needed."""
        if not self.system_prompt:
            return
        if any(
            message.role == "system" and message.content == self.system_prompt
            for message in self._history
        ):
            return
        self._history.insert(0, Message(role="system", content=self.system_prompt))

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)
    
    def clear_history(self):
        """清空历史记录"""
        self._history.clear()
    
    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self) -> str:
        provider = getattr(self.llm, "provider", self.llm.__class__.__name__)
        return f"Agent(name={self.name}, provider={provider})"
