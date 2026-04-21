# AGENTS.md

## 当前状态
- 这是一个早期学习脚手架项目。优先进行小型、可逆的编辑，避免大规模清理或抽象。
- 信任可执行源码而非文档。`README.md` 和 `main.py` 与代码库不同步：测试导入 `core.config.Settings`，但 `core/config.py` 当前定义的是 `Config`。

## 高价值路径
- `core/` 包含主要边界类型和抽象：Agent 基类、LLM 接口、配置、消息模型和异常。
- `agents/` 包含轻量级骨架 Agent；这些是提示模式示例，不是完整的工具调用实现。
- `tools/` 包含注册表、顺序链、异步执行器和内置工具。
- `memory/` 包含记忆子系统：工作记忆、情景记忆、语义记忆、感知记忆，以及 RAG 管道。
- `study/` 是实验性代码，可能遵循与主脚手架不同的假设。
- `.opencode/skills/` 包含仓库本地的可重用技能。任务匹配时重用或扩展这些技能。

## 已验证命令
- 使用 `.venv\Scripts\Activate.ps1` 激活项目虚拟环境
- 使用 `python -m unittest discover` 运行聚焦验证
- Ruff 配置在 `pyproject.toml` 中；需要时运行 `ruff check .`
- 测试记忆系统：`python test_memory_system.py`

## 本仓库工作规则
- 在假设文档或入口点最新之前，先用测试验证行为。
- 执行其他任务时，不要静默"修复"文档与代码之间的无关差异；除非用户要求清理，否则在最终响应中注明。
- 更改核心边界对象（如 `Message`）时，保持外部 OpenAI 兼容的有效载荷形状最小化，并为运行时验证和序列化添加测试。
- 避免将 `ReActAgent` 或内置工具视为生产就绪流程；除非明确实现缺失的执行循环，否则它们只是脚手架和占位符。
- 记忆系统是实验性的，包含多种存储后端（Qdrant、Neo4j、DocumentStore）和嵌入服务。

## 现有指令来源
- `Agents.md` 存在于仓库根目录作为旧的项目指导，但此 `AGENTS.md` 应被视为未来 OpenCode 会话的当前紧凑来源。
- 仓库本地技能参考：`.opencode/skills/SkillGeneration/SKILL.md`
- 消息边界设计的仓库本地技能：`.opencode/skills/message-model-design/SKILL.md`
