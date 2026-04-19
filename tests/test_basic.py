import unittest

from agents.simple_agent import SimpleAgent
from core.config import Settings
from core.llm import MockLLM


class TestBasicScaffold(unittest.TestCase):
    def test_settings_defaults(self) -> None:
        settings = Settings()
        settings.validate()
        self.assertEqual(settings.app_name, "Hello_Agents")

    def test_simple_agent_returns_mock_response(self) -> None:
        agent = SimpleAgent(llm=MockLLM())
        response = agent.run("hello")
        self.assertIn("hello", response.content)


if __name__ == "__main__":
    unittest.main()
