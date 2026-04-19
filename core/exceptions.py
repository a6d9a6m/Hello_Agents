"""Project-specific exception hierarchy."""


class HelloAgentsError(Exception):
    """Base exception for all project errors."""


class ConfigError(HelloAgentsError):
    """Raised when configuration is invalid or missing."""


class LLMError(HelloAgentsError):
    """Raised when LLM invocation fails."""


class ToolError(HelloAgentsError):
    """Raised when tool execution fails."""


class AgentError(HelloAgentsError):
    """Raised when agent execution fails."""
