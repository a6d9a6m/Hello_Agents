import unittest

from src.agents.plan_solve_agent import PlanAndSolveAgent
from src.agents.react_agent import ReActAgent
from src.agents.reflection_agent import ReflectionAgent
from src.core.llm import HelloAgentsLLM, MockLLM
from src.core.message import Message
from src.tools.base import Tool, ToolParameter, ToolResult
from src.tools.registry import ToolRegistry


class FakePatternLLM(HelloAgentsLLM):
    def generate(self, messages):
        last_message = messages[-1]
        content = last_message.content

        if content == "Create a concise plan only for the user's question. Do not solve it yet.":
            return Message(role="assistant", content="Plan: 1. Identify the key facts. 2. Answer concisely.")

        if content == "Using the user's question and the plan above, provide the final answer.":
            return Message(role="assistant", content="Final: The answer follows the plan.")

        if content == "Write a first draft answer to the user's question.":
            return Message(role="assistant", content="Draft: Initial answer.")

        if content == "Reflect on the draft for correctness, missing information, and clarity. Do not rewrite it yet.":
            return Message(role="assistant", content="Reflection: Add one missing detail and tighten wording.")

        if content == "Revise the draft using the reflection above and provide the final answer.":
            return Message(role="assistant", content="Final: Revised answer with the missing detail.")

        if content == "No tools are available. Use a ReAct-style approach internally, but return only the final answer.":
            return Message(role="assistant", content="Final: Direct ReAct-style answer without tools.")

        if content.startswith("Think in ReAct format"):
            return Message(
                role="assistant",
                content="Thought: I should use the tool.\nAction: lookup\nAction Input: weather in beijing",
            )

        if content == "Use the tool observation above to provide the final answer to the user.":
            tool_messages = [message for message in messages if message.role == "tool"]
            observation = tool_messages[-1].content if tool_messages else "no observation"
            return Message(role="assistant", content=f"Final: Based on observation -> {observation}")

        if content == "Provide the best final answer directly based on the conversation so far.":
            return Message(role="assistant", content="Final: Fallback answer.")

        return Message(role="assistant", content="Unhandled prompt")


class LookupTool(Tool):
    name = "lookup"
    description = "Look up a string value"
    parameters = [
        ToolParameter(name="input", param_type=str, description="Lookup input"),
    ]

    def execute(self, **validated):
        return ToolResult(tool_name=self.name, output=f"Observation for {validated['input']}")


class TestAgentPatterns(unittest.TestCase):
    def test_agent_str_works_with_mock_llm(self) -> None:
        agent = PlanAndSolveAgent(name="planner", llm=MockLLM())

        self.assertEqual(str(agent), "Agent(name=planner, provider=MockLLM)")

    def test_plan_and_solve_runs_two_stage_flow(self) -> None:
        agent = PlanAndSolveAgent(name="planner", llm=FakePatternLLM())

        response = agent.run("How should I answer this?")
        history = agent.get_history()

        self.assertEqual(response.content, "Final: The answer follows the plan.")
        self.assertEqual([message.role for message in history], ["user", "user", "assistant", "user", "assistant"])
        self.assertEqual(history[0].content, "How should I answer this?")
        self.assertIn("Plan:", history[2].content)
        self.assertIn("Final:", history[4].content)

    def test_reflection_runs_three_stage_flow(self) -> None:
        agent = ReflectionAgent(name="reflector", llm=FakePatternLLM())

        response = agent.run("Explain the topic.")
        history = agent.get_history()

        self.assertEqual(response.content, "Final: Revised answer with the missing detail.")
        self.assertEqual([message.role for message in history], ["user", "user", "assistant", "user", "assistant", "user", "assistant"])
        self.assertIn("Draft:", history[2].content)
        self.assertIn("Reflection:", history[4].content)
        self.assertIn("Final:", history[6].content)

    def test_react_without_tools_returns_direct_answer(self) -> None:
        agent = ReActAgent(name="react", llm=FakePatternLLM(), tool_registry=ToolRegistry())

        response = agent.run("Answer directly.")
        history = agent.get_history()

        self.assertEqual(response.content, "Final: Direct ReAct-style answer without tools.")
        self.assertEqual([message.role for message in history], ["user", "user", "assistant"])

    def test_react_with_tool_completes_single_tool_closure(self) -> None:
        registry = ToolRegistry([LookupTool()])
        agent = ReActAgent(name="react", llm=FakePatternLLM(), tool_registry=registry)

        response = agent.run("What is the weather?")
        history = agent.get_history()

        self.assertIn("Observation for weather in beijing", response.content)
        self.assertEqual([message.role for message in history], ["user", "user", "assistant", "tool", "user", "assistant"])
        self.assertEqual(history[3].metadata["tool_name"], "lookup")
        self.assertEqual(history[3].content, "Observation for weather in beijing")


if __name__ == "__main__":
    unittest.main()
