# Hello_Agents

一个面向 Agent 学习和实验的 Python 项目骨架。当前版本先把核心框架、常见 Agent 形态、工具系统和基础工程文件搭起来，方便你后续继续扩展真实的 LLM 接入、记忆、规划和工具调用链路。

## 目录结构

```text
Hello_Agents/
├── agents/
│   ├── plan_solve_agent.py
│   ├── react_agent.py
│   ├── reflection_agent.py
│   └── simple_agent.py
├── core/
│   ├── agent.py
│   ├── config.py
│   ├── exceptions.py
│   ├── llm.py
│   └── message.py
├── tools/
│   ├── async_executor.py
│   ├── base.py
│   ├── chain.py
│   ├── registry.py
│   └── builtin/
│       ├── calculator.py
│       └── search.py
├── tests/
│   └── test_basic.py
├── .env.example
├── .gitignore
├── main.py
└── pyproject.toml
```

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python main.py
python -m unittest discover
```

## 当前已包含的基础能力

- `core/`：统一的 Agent、LLM、消息、配置和异常抽象
- `agents/`：`SimpleAgent`、`ReActAgent`、`ReflectionAgent`、`PlanAndSolveAgent`
- `tools/`：工具基类、注册表、工具链、异步执行器、内置计算器和搜索占位工具
- `main.py`：一个最小可运行示例

## 后续建议

- 在 `core/llm.py` 中增加真实 OpenAI 或兼容接口实现
- 给 `ReActAgent` 增加工具调用协议与解析器
- 增加 `memory/`、`prompts/`、`examples/` 等目录
- 补充单元测试和端到端样例
