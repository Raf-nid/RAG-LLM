"""
Unit tests for CalculatorTool and its safe evaluator.

Tests are organised by:
1. Safe evaluator (_safe_eval) — arithmetic, edge cases, security
2. Input schema (CalculatorInput) — Pydantic validation
3. CalculatorTool.execute() — end-to-end, output format
4. CalculatorTool metadata — name, description, schema
"""

import pytest

from rag_assistant.tools.base import ToolError
from rag_assistant.tools.calculator import CalculatorInput, CalculatorTool, _safe_eval

# ---------------------------------------------------------------------------
# 1. _safe_eval
# ---------------------------------------------------------------------------


class TestSafeEval:
    def test_addition(self) -> None:
        assert _safe_eval("1 + 2") == 3.0

    def test_subtraction(self) -> None:
        assert _safe_eval("10 - 3") == 7.0

    def test_multiplication(self) -> None:
        assert _safe_eval("4 * 5") == 20.0

    def test_division(self) -> None:
        assert _safe_eval("10 / 4") == 2.5

    def test_power(self) -> None:
        assert _safe_eval("2 ** 10") == 1024.0

    def test_floor_division(self) -> None:
        assert _safe_eval("10 // 3") == 3.0

    def test_modulo(self) -> None:
        assert _safe_eval("10 % 3") == 1.0

    def test_unary_negative(self) -> None:
        assert _safe_eval("-5 + 10") == 5.0

    def test_unary_positive(self) -> None:
        assert _safe_eval("+5") == 5.0

    def test_parentheses(self) -> None:
        assert _safe_eval("(2 + 3) * 4") == 20.0

    def test_nested_parentheses(self) -> None:
        assert _safe_eval("((1 + 2) * (3 + 4))") == 21.0

    def test_float_literal(self) -> None:
        assert _safe_eval("3.14 * 2") == pytest.approx(6.28)

    def test_whitespace_stripped(self) -> None:
        assert _safe_eval("  10 + 5  ") == 15.0

    # --- Security: unsafe operations must raise ToolError ---

    def test_blocks_name_lookup(self) -> None:
        with pytest.raises(ToolError, match="Unsafe operation"):
            _safe_eval("x + 1")

    def test_blocks_function_call(self) -> None:
        with pytest.raises(ToolError, match="Unsafe operation"):
            _safe_eval("abs(-1)")

    def test_blocks_attribute_access(self) -> None:
        with pytest.raises(ToolError, match="Unsafe operation"):
            _safe_eval("(1).__class__")

    def test_blocks_import(self) -> None:
        with pytest.raises(ToolError, match="Unsafe operation"):
            _safe_eval("__import__('os')")

    def test_blocks_comparison(self) -> None:
        with pytest.raises(ToolError, match="Unsafe operation"):
            _safe_eval("1 == 1")

    def test_blocks_boolean_and(self) -> None:
        with pytest.raises(ToolError, match="Unsafe operation"):
            _safe_eval("True and False")

    # --- Division by zero ---

    def test_division_by_zero(self) -> None:
        with pytest.raises(ToolError, match="Division by zero"):
            _safe_eval("1 / 0")

    def test_floor_division_by_zero(self) -> None:
        with pytest.raises(ToolError, match="Division by zero"):
            _safe_eval("5 // 0")

    # --- Syntax error ---

    def test_syntax_error(self) -> None:
        with pytest.raises(ToolError, match="Invalid expression syntax"):
            _safe_eval("10 +* 5")

    def test_empty_expression_syntax_error(self) -> None:
        with pytest.raises(ToolError):
            _safe_eval("")


# ---------------------------------------------------------------------------
# 2. CalculatorInput schema
# ---------------------------------------------------------------------------


class TestCalculatorInput:
    def test_valid(self) -> None:
        inp = CalculatorInput(expression="1 + 1")
        assert inp.expression == "1 + 1"

    def test_empty_expression_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CalculatorInput(expression="")

    def test_missing_expression_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CalculatorInput()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 3. CalculatorTool.execute()
# ---------------------------------------------------------------------------


class TestCalculatorToolExecute:
    def setup_method(self) -> None:
        self.tool = CalculatorTool()

    def _args(self, expr: str) -> CalculatorInput:
        return CalculatorInput(expression=expr)

    def test_integer_result_no_decimal(self) -> None:
        assert self.tool.execute(self._args("6 * 7")) == "42"

    def test_float_result(self) -> None:
        result = self.tool.execute(self._args("10 / 3"))
        assert "3.33" in result  # at least 2 decimal places

    def test_large_number(self) -> None:
        assert self.tool.execute(self._args("1000 * 1000")) == "1000000"

    def test_negative_result(self) -> None:
        assert self.tool.execute(self._args("5 - 10")) == "-5"

    def test_power_of_two(self) -> None:
        assert self.tool.execute(self._args("2 ** 10")) == "1024"

    def test_complex_expression(self) -> None:
        # (100 + 50) * 2 / 3 = 100.0 exactly → integer output
        assert self.tool.execute(self._args("(100 + 50) * 2 / 3")) == "100"

    def test_error_propagated(self) -> None:
        with pytest.raises(ToolError):
            self.tool.execute(self._args("1 / 0"))

    def test_unsafe_expression_raises_tool_error(self) -> None:
        with pytest.raises(ToolError, match="Unsafe operation"):
            self.tool.execute(self._args("__import__('os').getcwd()"))


# ---------------------------------------------------------------------------
# 4. CalculatorTool metadata
# ---------------------------------------------------------------------------


class TestCalculatorToolMetadata:
    def setup_method(self) -> None:
        self.tool = CalculatorTool()

    def test_name(self) -> None:
        assert self.tool.name == "calculator"

    def test_description_non_empty(self) -> None:
        assert len(self.tool.description) > 0

    def test_schema_has_expression_field(self) -> None:
        schema = self.tool.to_tool_schema_params()
        assert "expression" in schema.get("properties", {})

    def test_schema_is_dict(self) -> None:
        schema = self.tool.to_tool_schema_params()
        assert isinstance(schema, dict)

    def test_repr(self) -> None:
        assert "calculator" in repr(self.tool)
