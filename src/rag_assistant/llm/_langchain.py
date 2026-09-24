"""
Shared LangChain conversion utilities.

Internal module — never import this from outside the ``llm`` package.
Both provider implementations use these helpers to convert between our
domain types and LangChain's message types.

Why separate from ``interface.py``?
-------------------------------------
``interface.py`` has no external dependencies — it is pure Python stdlib.
This module imports ``langchain_core``.  Keeping them separate prevents
a LangChain import whenever the interface types are used.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage as LCToolMessage

from .interface import ChatMessage, MessageRole, TokenUsage, ToolCallRequest, ToolSchema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def to_langchain_messages(messages: list[ChatMessage]) -> list[BaseMessage]:
    """
    Convert domain ``ChatMessage`` objects to LangChain message objects.

    Handles all four roles:
    - system    → SystemMessage
    - user      → HumanMessage
    - assistant → AIMessage (with or without tool_calls)
    - tool      → ToolMessage (tool execution result)

    The ``tool`` role and ``tool_call_id`` field are used after a tool has
    been executed to feed results back to the model in a follow-up call.

    Raises
    ------
    ValueError
        If a message role is not handled (guards against future additions).
    """
    result: list[BaseMessage] = []
    for msg in messages:
        if msg.role == MessageRole.system:
            result.append(SystemMessage(content=msg.content))

        elif msg.role == MessageRole.user:
            result.append(HumanMessage(content=msg.content))

        elif msg.role == MessageRole.assistant:
            if msg.tool_calls:
                # Assistant message that issued tool call requests.
                # LangChain needs the tool_calls in its own format.
                lc_tool_calls = [
                    {
                        "id": tc.call_id,
                        "name": tc.tool_name,
                        "args": tc.arguments,
                        "type": "tool_call",
                    }
                    for tc in msg.tool_calls
                ]
                result.append(AIMessage(content=msg.content, tool_calls=lc_tool_calls))
            else:
                result.append(AIMessage(content=msg.content))

        elif msg.role == MessageRole.tool:
            # Tool execution result.
            # ``tool_call_id`` must match the id from the preceding tool call request.
            result.append(
                LCToolMessage(
                    content=msg.content,
                    tool_call_id=msg.tool_call_id or "",
                )
            )

        else:  # pragma: no cover — exhaustive by StrEnum design
            raise ValueError(f"Unsupported message role: {msg.role!r}")

    return result


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


def parse_usage(response: AIMessage, provider_name: str = "unknown") -> TokenUsage:
    """
    Extract token counts from a LangChain ``AIMessage``.

    Returns zeros when ``usage_metadata`` is absent (streaming, some models)
    and logs a warning so operators can detect missing data without crashing.
    """
    meta = response.usage_metadata
    if meta is None:
        logger.warning(
            "usage_metadata was None for provider=%s; token counts unavailable",
            provider_name,
        )
        return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    return TokenUsage(
        prompt_tokens=meta["input_tokens"],
        completion_tokens=meta["output_tokens"],
        total_tokens=meta["total_tokens"],
    )


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------


def tool_schema_to_lc_dict(schema: ToolSchema) -> dict[str, Any]:
    """
    Convert a ``ToolSchema`` to the OpenAI/LangChain function-definition dict.

    LangChain's ``bind_tools`` and both the Groq and Ollama integrations
    accept this dict format directly (it mirrors the OpenAI tools API).

    The resulting dict looks like:
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "...",
            "parameters": {JSON Schema object}
        }
    }
    """
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        },
    }


def parse_tool_calls(response: AIMessage) -> tuple[ToolCallRequest, ...]:
    """
    Extract ``ToolCallRequest`` objects from a LangChain ``AIMessage``.

    Returns an empty tuple if the model responded directly without requesting
    any tools.  Each item in the tuple corresponds to one tool call request.
    """
    if not response.tool_calls:
        return ()
    return tuple(
        ToolCallRequest(
            call_id=tc["id"],
            tool_name=tc["name"],
            arguments=dict(tc["args"]),
        )
        for tc in response.tool_calls
    )
