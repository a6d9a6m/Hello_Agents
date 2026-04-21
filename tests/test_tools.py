import types
import unittest

from src.core.exceptions import ToolError
from src.tools.base import Tool, ToolParameter, ToolResult
from src.tools.builtin.calculator import CalculatorTool
from src.tools.builtin.search import SearchTool
from src.tools.registry import ToolRegistry


class ExampleTool(Tool):
    name = "example"
    description = "Example tool"
    parameters = [
        ToolParameter(name="text", param_type=str, description="Input text"),
        ToolParameter(name="count", param_type=int, description="Repeat count", required=False, default=1),
        ToolParameter(name="mode", param_type=str, description="Mode", required=False, default="plain", enum=["plain", "upper"]),
    ]

    def execute(self, **validated):
        text = validated["text"]
        if validated["mode"] == "upper":
            text = text.upper()
        return ToolResult(tool_name=self.name, output=text * validated["count"])


class TestToolParameter(unittest.TestCase):
    def test_validate_basic_types(self) -> None:
        self.assertEqual(ToolParameter("text", str).validate("hello"), "hello")
        self.assertEqual(ToolParameter("count", int).validate(3), 3)
        self.assertEqual(ToolParameter("ratio", float).validate(1.5), 1.5)
        self.assertEqual(ToolParameter("flag", bool).validate(True), True)
        self.assertEqual(ToolParameter("items", list).validate([1, 2]), [1, 2])
        self.assertEqual(ToolParameter("payload", dict).validate({"a": 1}), {"a": 1})

    def test_validate_rejects_invalid_bool_and_type(self) -> None:
        with self.assertRaises(ToolError):
            ToolParameter("flag", bool).validate("true")
        with self.assertRaises(ToolError):
            ToolParameter("count", int).validate(True)

    def test_required_default_and_enum(self) -> None:
        tool = ExampleTool()

        with self.assertRaises(ToolError):
            tool.run()

        with self.assertRaises(ToolError):
            tool.run(text="hi", mode="invalid")

        result = tool.run(text="hi")
        self.assertEqual(result.output, "hi")


class TestToolSchema(unittest.TestCase):
    def test_to_openai_schema(self) -> None:
        schema = ExampleTool().to_openai_schema()

        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "example")
        self.assertEqual(schema["function"]["parameters"]["type"], "object")
        self.assertEqual(schema["function"]["parameters"]["properties"]["text"]["type"], "string")
        self.assertEqual(schema["function"]["parameters"]["properties"]["count"]["default"], 1)
        self.assertEqual(schema["function"]["parameters"]["required"], ["text"])


class TestToolRegistry(unittest.TestCase):
    def test_register_get_list_and_execute(self) -> None:
        registry = ToolRegistry()
        tool = ExampleTool()

        registry.register(tool)

        self.assertIs(registry.get("example"), tool)
        self.assertEqual(registry.list_tools(), ["example"])
        result = registry.execute("example", text="ok", count=2)
        self.assertEqual(result.output, "okok")
        self.assertEqual(registry.list_definitions()[0]["name"], "example")
        self.assertEqual(registry.list_openai_schemas()[0]["function"]["name"], "example")

    def test_discover_from_module(self) -> None:
        module = types.ModuleType("fake_tools")
        discovered_tool = type(
            "DiscoveredTool",
            (Tool,),
            {
                "__module__": "fake_tools",
                "name": "discovered",
                "description": "Discovered tool",
                "parameters": [],
                "execute": lambda self, **validated: ToolResult(tool_name=self.name, output="done"),
            },
        )
        setattr(module, "DiscoveredTool", discovered_tool)

        registry = ToolRegistry()
        discovered = registry.discover_from_module(module)

        self.assertEqual([tool.name for tool in discovered], ["discovered"])
        self.assertEqual(registry.list_tools(), ["discovered"])


class TestBuiltinTools(unittest.TestCase):
    def test_calculator_executes(self) -> None:
        result = CalculatorTool().run(expression="1 + 2 * 3")
        self.assertTrue(result.success)
        self.assertEqual(result.output, 7)

    def test_search_executes(self) -> None:
        result = SearchTool().run(query="python")
        self.assertTrue(result.success)
        self.assertIn("python", result.output)


if __name__ == "__main__":
    unittest.main()
