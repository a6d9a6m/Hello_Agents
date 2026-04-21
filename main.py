"""Example entrypoint for the Hello_Agents scaffold."""

from src.core.config import Config
from src.core.llm import MockLLM
from src.agents.simple_agent import SimpleAgent


def main() -> None:
    config = Config.from_env()

    agent = SimpleAgent(name="hello_agent", llm=MockLLM())
    response = agent.run("你好，介绍一下 Hello_Agents 的基础结构。")
    print(f"[Hello_Agents] {response.content}")


if __name__ == "__main__":
    main()
