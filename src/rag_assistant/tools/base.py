"""
Tool base types.

Defines the abstract contract that every tool must satisfy, the result
container, and the error type.

Design notes
------------
- Tools are pure Python.  No LangChain, no LLM, no HTTP.
- ``BaseTool`` is an ABC.  Concrete tools define ``name``, ``description``,
  ``input_schema``, and ``execute()``.
- The application is responsible for validating arguments before calling
  ``execute()``.  ``ToolRegistry`` (in ``registry.py``) handles this.
- ``execute()`` must raise ``ToolError`` for expected failures (bad input,
  domain errors).  Unexpected exceptions propagate to the registry.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """
    Raised by a tool when execution fails for an expected reason.

    Use this for domain errors such as division by zero, a resource not found,
    or an unsupported operation.  Do NOT use it for programming errors (let
    those propagate as regular exceptions).

    The ``ToolRegistry`` catches ``ToolError`` and returns it as a
    ``ToolResult(is_error=True)`` rather than allowing it to crash the caller.
    """

    def __init__(self, message: str, tool_name: str = "") -> None:
        super().__init__(message)
        self.tool_name = tool_name


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    """
    The outcome of executing a tool.

    Fields
    ------
    call_id
        Echo of the ``ToolCallRequest.call_id`` from the model.  Used to
        correlate results with requests in multi-tool conversations.
    tool_name
        Name of the tool that was called.
    content
        String output from the tool.  May be plain text, JSON, or a short
        error message when ``is_error=True``.
    is_error
        When True, the tool failed.  ``content`` contains the error description.
        The model will receive this as a tool result and can generate a suitable
        response to the user.
    """

    call_id: str
    tool_name: str
    content: str
    is_error: bool = field(default=False)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Subclasses must define:
    - ``name``         : class variable, unique tool identifier
    - ``description``  : class variable, human-readable description for the LLM
    - ``input_schema`` : class variable, Pydantic model for argument validation
    - ``execute()``    : method that performs the actual work

    The ``input_schema`` is used both for argument validation and for
    generating the JSON Schema that is sent to the LLM.

    Example
    -------
    class EchoInput(BaseModel):
        text: str

    class EchoTool(BaseTool):
        name = "echo"
        description = "Returns the input text unchanged."
        input_schema = EchoInput

        def execute(self, args: BaseModel) -> str:
            assert isinstance(args, EchoInput)
            return args.text
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]

    @abstractmethod
    def execute(self, args: BaseModel) -> str:
        """
        Execute the tool with pre-validated arguments.

        Parameters
        ----------
        args:
            A validated instance of ``self.input_schema``.  Callers (the
            ``ToolRegistry``) guarantee that ``args`` is an instance of the
            correct type.

        Returns
        -------
        str
            The tool's result as a string.  May be plain text or JSON.

        Raises
        ------
        ToolError
            For expected, domain-level failures.  The registry will catch
            this and return it as ``ToolResult(is_error=True)``.
        """
        ...

    def to_tool_schema_params(self) -> dict[str, Any]:
        """
        Return the JSON Schema parameters dict for use in a ``ToolSchema``.

        This is the ``parameters`` field that the LLM receives when deciding
        whether and how to call this tool.
        """
        return self.input_schema.model_json_schema()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
