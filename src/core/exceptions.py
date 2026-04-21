"""项目特定的异常层次结构"""


class HelloAgentsError(Exception):
    """所有项目错误的基础异常"""


class ConfigError(HelloAgentsError):
    """配置无效或缺失时抛出"""


class LLMError(HelloAgentsError):
    """LLM调用失败时抛出"""


class ToolError(HelloAgentsError):
    """工具执行失败时抛出"""


class AgentError(HelloAgentsError):
    """Agent执行失败时抛出"""
