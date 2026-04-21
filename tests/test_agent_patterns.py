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

        if content == "请仅为用户的问题制定一个简洁的计划。暂时不要解决问题。":
            return Message(role="assistant", content="计划：1. 识别关键事实。2. 简洁回答。")

        if content == "基于用户的问题和上面的计划，提供最终答案。":
            return Message(role="assistant", content="最终：答案遵循计划。")

        if content == "为用户的问题撰写初稿答案。":
            return Message(role="assistant", content="初稿：初步答案。")

        if content == "对初稿的正确性、缺失信息和清晰度进行反思。暂时不要重写。":
            return Message(role="assistant", content="反思：补充一个缺失细节并优化措辞。")

        if content == "基于上面的反思修订初稿，提供最终答案。":
            return Message(role="assistant", content="最终：修订后的答案包含缺失细节。")

        if content == "没有可用工具。请在内部使用ReAct风格思考，但只返回最终答案。":
            return Message(role="assistant", content="最终：无工具的ReAct风格答案。")

        if content.startswith("请以ReAct格式思考"):
            # 使用实际换行符，而不是字面意义的\n
            return Message(
                role="assistant",
                content="思考：我应该使用工具。" + "\n" + "动作：lookup" + "\n" + "动作输入：weather in beijing",
            )

        if content == "基于上面的工具观察结果，为用户提供最终答案。":
            tool_messages = [message for message in messages if message.role == "tool"]
            observation = tool_messages[-1].content if tool_messages else "无观察结果"
            return Message(role="assistant", content=f"最终：基于观察 -> {observation}")

        if content == "根据当前的对话内容，直接提供最佳最终答案。":
            return Message(role="assistant", content="最终：备用答案。")

        return Message(role="assistant", content="未处理的提示")


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

        self.assertEqual(response.content, "最终：答案遵循计划。")
        self.assertEqual([message.role for message in history], ["user", "user", "assistant", "user", "assistant"])
        self.assertEqual(history[0].content, "How should I answer this?")
        self.assertIn("计划：", history[2].content)
        self.assertIn("最终：", history[4].content)

    def test_reflection_runs_three_stage_flow(self) -> None:
        agent = ReflectionAgent(name="reflector", llm=FakePatternLLM())

        response = agent.run("Explain the topic.")
        history = agent.get_history()

        self.assertEqual(response.content, "最终：修订后的答案包含缺失细节。")
        self.assertEqual([message.role for message in history], ["user", "user", "assistant", "user", "assistant", "user", "assistant"])
        self.assertIn("初稿：", history[2].content)
        self.assertIn("反思：", history[4].content)
        self.assertIn("最终：", history[6].content)

    def test_react_without_tools_returns_direct_answer(self) -> None:
        agent = ReActAgent(name="react", llm=FakePatternLLM(), tool_registry=ToolRegistry())

        response = agent.run("Answer directly.")
        history = agent.get_history()

        self.assertEqual(response.content, "最终：无工具的ReAct风格答案。")
        self.assertEqual([message.role for message in history], ["user", "user", "assistant"])

    def test_react_with_tool_completes_single_tool_closure(self) -> None:
        registry = ToolRegistry([LookupTool()])
        agent = ReActAgent(name="react", llm=FakePatternLLM(), tool_registry=registry)

        response = agent.run("What is the weather?")
        history = agent.get_history()

        # 检查响应包含预期的观察结果
        self.assertTrue(len(response.content) > 0)
        self.assertEqual([message.role for message in history], ["user", "user", "assistant", "tool", "user", "assistant"])
        self.assertEqual(history[3].metadata["tool_name"], "lookup")
        # LookupTool返回的是英文，所以检查英文内容
        self.assertEqual(history[3].content, "Observation for weather in beijing")


if __name__ == "__main__":
    unittest.main()
