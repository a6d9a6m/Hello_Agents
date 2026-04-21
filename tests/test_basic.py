import unittest
from datetime import timezone
from pydantic import ValidationError

from src.agents.simple_agent import SimpleAgent
from src.core.config import Config
from src.core.llm import MockLLM
from src.core.message import Message


class TestBasicScaffold(unittest.TestCase):
    def test_config_defaults(self) -> None:
        config = Config()
        self.assertEqual(config.default_model, "gpt-3.5-turbo")
        self.assertEqual(config.default_provider, "openai")
        self.assertEqual(config.temperature, 0.7)

    def test_simple_agent_returns_mock_response(self) -> None:
        agent = SimpleAgent(name="test_agent", llm=MockLLM())
        response = agent.run("hello")
        self.assertIn("hello", response.content)

    def test_message_tracks_internal_fields(self) -> None:
        message = Message(role="user", content="hello", metadata={"source": "test"})

        self.assertEqual(message.metadata["source"], "test")
        self.assertEqual(message.timestamp.tzinfo, timezone.utc)

    def test_message_to_dict_matches_openai_shape(self) -> None:
        message = Message(role="assistant", content="world", metadata={"trace_id": "123"})

        self.assertEqual(
            message.to_dict(),
            {
                "role": "assistant",
                "content": "world",
            },
        )

    def test_message_kwargs_constructor_applies_defaults(self) -> None:
        message = Message(role="system", content="rules")

        self.assertEqual(message.role, "system")
        self.assertEqual(message.content, "rules")
        self.assertEqual(message.metadata, {})
        self.assertEqual(message.timestamp.tzinfo, timezone.utc)

    def test_message_str_uses_role_and_content(self) -> None:
        message = Message(role="tool", content="result")

        self.assertEqual(str(message), "[tool] result")

    def test_message_rejects_invalid_role(self) -> None:
        with self.assertRaises(ValidationError):
            Message(role="developer", content="rules")


if __name__ == "__main__":
    unittest.main()
