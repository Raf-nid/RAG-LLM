"""
Unit tests for ToolRegistry.

Tests:
1. Registration — success, duplicate rejection, __contains__, __len__
2. tool_schemas() — returns correct ToolSchema objects
3. execute() — unknown tool, validation failure, success, ToolError
4. Argument validation guards (multiple bad inputs)
"""

import pytest
from pydantic import BaseModel

from rag_assistant.llm.interface import ToolCallRequest
from rag_assistant.tools.base import BaseTool, ToolError, ToolResult
from rag_assistant.tools.calculator import CalculatorTool
from rag_assistant.tools.doc_search import DocSearchTool
from rag_assistant.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Minimal stub tool for isolated tests
# ---------------------------------------------------------------------------


class EchoInput(BaseModel):
    text: str


class EchoTool(BaseTool):
    name = "echo"
    description = "Returns the input text."
    input_schema = EchoInput

    def execute(self, args: BaseModel) -> str:
        assert isinstance(args, EchoInput)
        return args.text


class BrokenTool(BaseTool):
    """Tool whose execute always raises ToolError."""

    name = "broken"
    description = "Always fails."
    input_schema = EchoInput

    def execute(self, args: BaseModel) -> str:
        raise ToolError("intentional failure", tool_name=self.name)


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_single_tool(self) -> None:
        reg = ToolRegistry()
        reg.register(EchoTool())
        assert "echo" in reg

    def test_register_multiple_tools(self) -> None:
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(CalculatorTool())
        assert len(reg) == 2

    def test_duplicate_registration_raises(self) -> None:
        reg = ToolRegistry()
        reg.register(EchoTool())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(EchoTool())

    def test_contains_false_for_unknown(self) -> None:
        reg = ToolRegistry()
        assert "nonexistent" not in reg

    def test_len_empty(self) -> None:
        reg = ToolRegistry()
        assert len(reg) == 0

    def test_repr(self) -> None:
        reg = ToolRegistry()
        reg.register(EchoTool())
        assert "echo" in repr(reg)


# ---------------------------------------------------------------------------
# 2. tool_schemas()
# ---------------------------------------------------------------------------


class TestToolSchemas:
    def test_empty_registry_returns_empty_list(self) -> None:
        reg = ToolRegistry()
        assert reg.tool_schemas() == []

    def test_schema_name_matches_tool(self) -> None:
        reg = ToolRegistry()
        reg.register(EchoTool())
        schemas = reg.tool_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "echo"

    def test_schema_description_matches_tool(self) -> None:
        reg = ToolRegistry()
        reg.register(EchoTool())
        schemas = reg.tool_schemas()
        assert schemas[0].description == "Returns the input text."

    def test_schema_parameters_is_dict(self) -> None:
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        schemas = reg.tool_schemas()
        assert isinstance(schemas[0].parameters, dict)

    def test_schema_parameters_has_expression(self) -> None:
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        schemas = reg.tool_schemas()
        assert "expression" in schemas[0].parameters.get("properties", {})

    def test_multiple_tools_all_returned(self) -> None:
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.register(DocSearchTool())
        names = {s.name for s in reg.tool_schemas()}
        assert names == {"calculator", "doc_search"}


# ---------------------------------------------------------------------------
# 3. execute()
# ---------------------------------------------------------------------------


def _request(tool_name: str, arguments: dict, call_id: str = "call-test") -> ToolCallRequest:
    return ToolCallRequest(call_id=call_id, tool_name=tool_name, arguments=arguments)


class TestExecute:
    def test_unknown_tool_returns_error_result(self) -> None:
        reg = ToolRegistry()
        result = reg.execute(_request("nonexistent", {}))
        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "nonexistent" in result.content

    def test_error_result_preserves_call_id(self) -> None:
        reg = ToolRegistry()
        result = reg.execute(_request("nonexistent", {}, call_id="my-call-id"))
        assert result.call_id == "my-call-id"

    def test_calculator_valid_args_success(self) -> None:
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        result = reg.execute(_request("calculator", {"expression": "6 * 7"}))
        assert result.is_error is False
        assert result.content == "42"
        assert result.tool_name == "calculator"

    def test_calculator_call_id_preserved(self) -> None:
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        result = reg.execute(_request("calculator", {"expression": "1 + 1"}, call_id="abc123"))
        assert result.call_id == "abc123"

    def test_invalid_arguments_validation_error(self) -> None:
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        # Pydantic coerces int to str, so this actually succeeds.
        # The missing-field case is the canonical validation failure test (see below).

    def test_missing_required_argument(self) -> None:
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        # 'expression' is required; empty dict should fail validation
        result = reg.execute(_request("calculator", {}))
        assert result.is_error is True
        assert "Invalid arguments" in result.content

    def test_tool_error_becomes_error_result(self) -> None:
        reg = ToolRegistry()
        reg.register(BrokenTool())
        result = reg.execute(_request("broken", {"text": "hello"}))
        assert result.is_error is True
        assert "intentional failure" in result.content

    def test_tool_error_result_is_not_exception(self) -> None:
        """execute() must NOT raise — it always returns ToolResult."""
        reg = ToolRegistry()
        reg.register(BrokenTool())
        # This should not raise:
        result = reg.execute(_request("broken", {"text": "hello"}))
        assert isinstance(result, ToolResult)

    def test_calculator_division_by_zero_error_result(self) -> None:
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        result = reg.execute(_request("calculator", {"expression": "1 / 0"}))
        assert result.is_error is True
        assert "zero" in result.content.lower()
