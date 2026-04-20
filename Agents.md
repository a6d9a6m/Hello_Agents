# Agents.md

## 自我更新
在你每次执行完任务的时候
如果CLAUDE.md新增一行内容，带来的TOKEN消耗期望更小
或者如果CLAUDE.md删除一行内容，带来的TOKEN消耗期望更小
就向人工提出请求修改agents.md，并解释详细原因

## Project status
This project is in early development. Architecture, folder structure, and APIs may change.
Prefer small, reversible changes over large rewrites.

## Product goal

This project aims to build: 一个人要搭建的agents示范

## 项目架构和技术栈
目前项目是对于hello_agents库的学习，主体是一个自己构建的类似hello_agents的骨架
### 目录目前结构
```text
Hello_Agents/
├── agents/          # 各类 Agent 实现
├── core/            # 核心抽象：Agent、LLM、Message、Config、Exception
├── study/           # 对于一些类的重写学习尝试
├── tools/           # 工具基类、注册表、执行器、内置工具
├── tests/           # 基础测试
├── main.py          # 最小运行入口
├── pyproject.toml   # 项目元信息与开发工具配置
└── README.md        # 项目说明
```


## Coding conventions
Keep implementations simple and explicit.
Avoid premature abstraction.
Prefer readable code over clever code.
When adding a new pattern, use it consistently in the touched area.
Do not silently change public APIs or data formats.

## Architecture preferences
Keep business logic separate from UI/components/routes where practical.
Avoid large files; split by responsibility when a file becomes hard to scan.
Prefer composition over inheritance.
Keep side effects close to boundaries: API calls, database access, filesystem access.


## 常用命令

### 环境初始化
项目使用了.venv的虚拟环境
```powershell
.venv\Scripts\Activate.ps1
```

## Do not
Do not add heavy dependencies casually.
Do not introduce global state unless necessary.
Do not create broad abstractions for hypothetical future use cases.
Do not overwrite existing work without checking context.

## Git and change discipline
Make focused changes.
Avoid unrelated formatting churn.
Do not rename files or restructure folders unless needed for the task.
Explain non-obvious decisions in the final response.