"""Message models shared across agents and LLMs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel


MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """Represents a single turn in a conversation."""

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
        """Convert the internal message model into the OpenAI chat format."""

        return {
            "role": self.role,
            "content": self.content,
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
