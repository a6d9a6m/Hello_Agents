"""Example entrypoint for the Hello_Agents scaffold."""

from core.config import Settings
from core.llm import MockLLM
from agents.simple_agent import SimpleAgent


def main() -> None:
    settings = Settings.from_env()
    settings.validate()

    agent = SimpleAgent(llm=MockLLM())
    response = agent.run("你好，介绍一下 Hello_Agents 的基础结构。")
    print(f"[{settings.app_name}] {response.content}")


if __name__ == "__main__":
    main()
