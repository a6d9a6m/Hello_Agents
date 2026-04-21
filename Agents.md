# AGENTS.md

## Current State
- This repo is an early-stage learning scaffold. Prefer small, reversible edits over broad cleanup or abstraction.
- Trust executable sources over prose. `README.md` and `main.py` are currently out of sync with the codebase: tests import `core.config.Settings`, but `core/config.py` currently defines `Config` instead.

## High-Value Paths
- `core/` holds the main boundary types and abstractions: agent base class, LLM interface, config, message model, and exceptions.
- `agents/` contains lightweight skeleton agents; these are prompt-pattern examples, not full tool-calling implementations.
- `tools/` contains the registry, sequential chain, async executor, and placeholder builtin tools.
- `study/` is experimental code and may follow different assumptions than the main scaffold.
- `.opencode/skills/` contains repo-local reusable skills. Reuse or extend these when the task matches.

## Verified Commands
- Activate the project virtualenv with `.venv\Scripts\Activate.ps1`
- Run focused verification with `python -m unittest discover`
- Ruff is configured in `pyproject.toml`; if needed, run `ruff check .`

## Working Rules For This Repo
- Verify behavior with tests before assuming docs or entrypoints are current.
- Do not silently “fix” unrelated drift between docs and code while doing another task; note it in the final response unless the user asked for that cleanup.
- When changing core boundary objects such as `Message`, keep the external OpenAI-compatible payload shape minimal and add tests for both runtime validation and serialization.
- Avoid treating `ReActAgent` or builtin tools as production-ready flows; they are scaffolds and placeholders unless you explicitly implement the missing execution loop.

## Existing Instruction Sources
- `Agents.md` exists in the repo root as older project guidance, but this `AGENTS.md` should be treated as the current compact source for future OpenCode sessions.
- Repo-local skill reference: `.opencode/skills/SkillGeneration/SKILL.md`
- Repo-local skill for message boundary design: `.opencode/skills/message-model-design/SKILL.md`
