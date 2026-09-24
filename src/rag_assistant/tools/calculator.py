"""
Calculator tool.

Evaluates arithmetic expressions safely using Python's ``ast`` module.

Security model
--------------
The expression is parsed into an AST and every node is checked against a
strict allowlist before evaluation.  Only these node types are permitted:

    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv
    ast.USub, ast.UAdd

Any other node (name lookups, function calls, attribute access, comparisons,
boolean operations, imports, etc.) raises ``ToolError`` before any evaluation.
The ``eval`` call runs in a completely empty namespace (``__builtins__={}``),
so even if the allowlist check somehow passed an unsafe node, it would fail
at runtime rather than executing arbitrary code.

This approach is intentionally conservative.  If you need ``sqrt`` or
trigonometric functions, add them as explicit named tools rather than
extending the expression evaluator.
"""

import ast
from typing import Any

from pydantic import BaseModel, Field

from .base import BaseTool, ToolError

# The complete set of AST node types that are safe to evaluate.
_SAFE_NODES: frozenset[type[ast.AST]] = frozenset(
    {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        # Binary operators
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.FloorDiv,
        # Unary operators
        ast.USub,
        ast.UAdd,
    }
)


def _safe_eval(expression: str, tool_name: str = "calculator") -> float:
    """
    Evaluate an arithmetic expression and return the result as float.

    Parameters
    ----------
    expression:
        A mathematical expression string, e.g. ``"(10 + 5) * 2"`` or
        ``"2 ** 10"``.
    tool_name:
        Passed to ``ToolError`` for context.

    Raises
    ------
    ToolError
        If the expression has syntax errors, contains unsafe operations,
        causes a ZeroDivisionError, or produces a non-numeric result.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(
            f"Invalid expression syntax: {exc.msg}",
            tool_name=tool_name,
        ) from exc

    # Walk every node and reject anything outside the allowlist.
    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in _SAFE_NODES:
            raise ToolError(
                f"Unsafe operation in expression: '{node_type.__name__}'. "
                "Only arithmetic operators and numeric literals are allowed.",
                tool_name=tool_name,
            )

    # Evaluate in a fully empty namespace — no builtins, no variables.
    empty_ns: dict[str, Any] = {}
    try:
        result = eval(
            compile(tree, "<calculator>", "eval"),
            {"__builtins__": {}},
            empty_ns,
        )
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero.", tool_name=tool_name) from exc
    except Exception as exc:
        raise ToolError(
            f"Evaluation failed: {exc}",
            tool_name=tool_name,
        ) from exc

    try:
        return float(result)
    except (TypeError, ValueError) as exc:
        raise ToolError(
            f"Expression did not produce a numeric result: {result!r}",
            tool_name=tool_name,
        ) from exc


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class CalculatorInput(BaseModel):
    """Input arguments for the calculator tool."""

    expression: str = Field(
        description=(
            "A mathematical expression to evaluate. "
            "Supports +, -, *, /, ** (power), // (floor division), % (modulo). "
            "Examples: '(10 + 5) * 2', '2 ** 10', '100 / 4'."
        ),
        min_length=1,
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class CalculatorTool(BaseTool):
    """
    Arithmetic expression evaluator.

    Evaluates pure mathematical expressions.  Does NOT support functions
    (sin, sqrt, etc.) or variables.  Use for numeric calculations only.
    """

    name = "calculator"
    description = (
        "Evaluates a mathematical expression and returns the numeric result. "
        "Supports: addition (+), subtraction (-), multiplication (*), "
        "division (/), power (**), floor division (//), modulo (%). "
        "Does NOT support functions (sin, sqrt, etc.) or variables. "
        "Example inputs: '(10 + 5) * 2', '2 ** 10', '100 / 4'."
    )
    input_schema = CalculatorInput

    def execute(self, args: BaseModel) -> str:
        """
        Evaluate the expression and return the result as a string.

        Whole-number results are returned without a decimal point.
        Float results use Python's default float representation.
        """
        assert isinstance(args, CalculatorInput)
        value = _safe_eval(args.expression, tool_name=self.name)

        # Return integers cleanly (no ".0" suffix).
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return str(value)
