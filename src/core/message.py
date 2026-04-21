"""跨Agent和LLM共享的消息模型"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel


MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """表示对话中的单轮交互"""

    content: str
    role: MessageRole
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None

    def __init__(self, content: str, role: MessageRole, **kwargs: Any) -> None:
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now(timezone.utc)),
            metadata=kwargs.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, str]:
        """将内部消息模型转换为OpenAI聊天格式"""

        return {
            "role": self.role,
            "content": self.content,
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
