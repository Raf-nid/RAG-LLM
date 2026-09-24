"""
Tool registry.

Responsibilities:
1. Hold the set of registered tools.
2. Expose ``ToolSchema`` objects for every registered tool (passed to the LLM).
3. Validate tool call arguments before execution using Pydantic.
4. Execute the correct tool and return a ``ToolResult``.
5. Catch ``ToolError`` and surface it as a non-crashing ``ToolResult(is_error=True)``.

Security note
-------------
Argument validation happens here, not inside each tool.  ``ToolRegistry.execute``
calls ``tool.input_schema.model_validate(raw_args)`` before calling
``tool.execute(validated_args)``.  This guarantees that every tool receives
a properly typed Pydantic model — the tool's ``execute()`` method can assert
the type rather than re-validate.

Never pass model-generated arguments directly to application code without
validation.  The model may produce incorrect types, missing fields, or
extra fields.
"""

import logging
from typing import Any

from pydantic import ValidationError

from rag_assistant.llm.interface import ToolCallRequest, ToolSchema

from .base import BaseTool, ToolError, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry of available tools.

    Usage
    -----
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(DocSearchTool())

    # Get schemas to send to the LLM:
    schemas = registry.tool_schemas()

    # Execute a tool call request from the LLM:
    result = registry.execute(tool_call_request)
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance.

        Raises
        ------
        ValueError
            If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. Each tool name must be unique."
            )
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def tool_schemas(self) -> list[ToolSchema]:
        """
        Return ``ToolSchema`` objects for all registered tools.

        These are passed to the LLM provider before inference so the model
        knows which tools are available.
        """
        return [
            ToolSchema(
                name=tool.name,
                description=tool.description,
                parameters=tool.to_tool_schema_params(),
            )
            for tool in self._tools.values()
        ]

    def execute(self, request: ToolCallRequest) -> ToolResult:
        """
        Validate arguments and execute a tool call request.

        Steps:
        1. Look up the tool by name.
        2. Validate ``request.arguments`` using the tool's Pydantic input schema.
        3. Call ``tool.execute(validated_args)``.
        4. Return a ``ToolResult`` with the output or error.

        The caller is never responsible for catching ``ToolError`` — this
        method always returns a ``ToolResult``, never raises.

        Parameters
        ----------
        request:
            A ``ToolCallRequest`` as produced by the model.  ``call_id``
            is echoed back in the result so it can be matched to the request.

        Returns
        -------
        ToolResult
            ``is_error=False`` on success.
            ``is_error=True`` when the tool name is unknown, arguments are
            invalid, or the tool raised ``ToolError``.
        """
        tool_name = request.tool_name

        # Step 1: Resolve the tool.
        tool = self._tools.get(tool_name)
        if tool is None:
            known = ", ".join(self._tools) or "(none)"
            logger.warning("Unknown tool requested: '%s'. Known: %s", tool_name, known)
            return ToolResult(
                call_id=request.call_id,
                tool_name=tool_name,
                content=f"Unknown tool '{tool_name}'. Available tools: {known}.",
                is_error=True,
            )

        # Step 2: Validate arguments.
        try:
            validated_args = tool.input_schema.model_validate(request.arguments)
        except ValidationError as exc:
            logger.warning("Tool '%s' received invalid arguments: %s", tool_name, exc)
            return ToolResult(
                call_id=request.call_id,
                tool_name=tool_name,
                content=_format_validation_error(exc),
                is_error=True,
            )

        # Step 3: Execute.
        try:
            content = tool.execute(validated_args)
            logger.debug("Tool '%s' succeeded: %s", tool_name, content[:80])
            return ToolResult(
                call_id=request.call_id,
                tool_name=tool_name,
                content=content,
                is_error=False,
            )
        except ToolError as exc:
            logger.warning("Tool '%s' raised ToolError: %s", tool_name, exc)
            return ToolResult(
                call_id=request.call_id,
                tool_name=tool_name,
                content=str(exc),
                is_error=True,
            )

    def __contains__(self, tool_name: str) -> bool:
        """Support ``"calculator" in registry``."""
        return tool_name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        names = list(self._tools)
        return f"ToolRegistry(tools={names!r})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_validation_error(exc: ValidationError) -> str:
    """Format a Pydantic validation error into a readable one-line string."""
    errors: list[Any] = exc.errors(include_url=False)
    parts = []
    for err in errors:
        loc = " → ".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "Invalid arguments: " + "; ".join(parts)
