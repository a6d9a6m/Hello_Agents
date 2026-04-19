"""Configuration loading utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from core.exceptions import ConfigError


def load_env_file(env_file: str | Path) -> None:
    """Load key/value pairs from a simple .env file into the environment."""

    env_path = Path(env_file)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(slots=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_name: str = "Hello_Agents"
    environment: str = "development"
    default_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    debug: bool = True
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Settings":
        load_env_file(env_file)

        extra = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("HELLO_AGENTS_")
        }
        return cls(
            app_name=os.getenv("APP_NAME", "Hello_Agents"),
            environment=os.getenv("APP_ENV", "development"),
            default_model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            debug=os.getenv("DEBUG", "true").lower() == "true",
            extra=extra,
        )

    def validate(self) -> None:
        if not self.default_model:
            raise ConfigError("DEFAULT_MODEL must not be empty.")
