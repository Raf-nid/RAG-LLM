"""
Tool-calling orchestration (single-round).

``run_with_tools`` implements one complete tool-calling round:

1. Send user messages to the LLM with tool schemas attached.
2. If the model requests tools: execute them, feed results back, ask again.
3. If the model responds directly: return immediately.

This is NOT an agent loop.  It executes at most one round of tool calls.
For multi-step reasoning and looping behaviour, use LangGraph.

Why not LangGraph yet?
----------------------
LangGraph is the right choice when:
- The model may need multiple rounds of tool use.
- Routing decisions (which sub-graph to enter) must be explicit.
- State must persist across user turns.
- Retries and fallback paths need to be defined.

Right now we have a single user query, two tools, and one possible round
of tool execution.  A simple function is sufficient and more transparent.

Return value
------------
``LLMResponse`` — the model's final text response.

Both providers (Groq, Ollama) are supported transparently because
``chat_with_tools`` is part of ``LLMProviderProtocol``.
"""

from __future__ import annotations

import logging

from rag_assistant.llm.interface import (
    ChatMessage,
    LLMProviderProtocol,
    LLMResponse,
    MessageRole,
    ToolCallingResponse,
)

from .registry import ToolRegistry

logger = logging.getLogger(__name__)


def run_with_tools(
    provider: LLMProviderProtocol,
    registry: ToolRegistry,
    messages: list[ChatMessage],
) -> LLMResponse:
    """
    Execute one round of tool-augmented inference.

    Parameters
    ----------
    provider:
        Any ``LLMProviderProtocol`` implementation (Groq, Ollama, or a mock).
    registry:
        Registry containing all available tools.
    messages:
        The conversation so far (typically [system, user]).

    Returns
    -------
    LLMResponse
        The model's final text response after tool execution (if any).

    Notes
    -----
    If the model issues no tool calls (direct response), this function
    immediately returns an ``LLMResponse`` built from the ``ToolCallingResponse``.
    If the model issues tool calls, all are executed, results are appended
    to the message list, and the model is called once more for a final answer.
    """
    tool_schemas = registry.tool_schemas()
    logger.debug("run_with_tools — %d tool(s) available", len(tool_schemas))

    # --- Round 1: send user messages + tool schemas ---
    tc_response: ToolCallingResponse = provider.chat_with_tools(messages, tool_schemas)

    if not tc_response.has_tool_calls:
        # Model responded directly — no tool calls needed.
        logger.debug("run_with_tools — model responded directly (no tool calls)")
        return _tc_to_llm_response(tc_response)

    logger.debug("run_with_tools — model issued %d tool call(s)", len(tc_response.tool_calls))

    # --- Execute all tool calls ---
    conversation: list[ChatMessage] = list(messages)

    # Append the assistant message that contains the tool call requests.
    conversation.append(
        ChatMessage(
            role=MessageRole.assistant,
            content=tc_response.content or "",
            tool_calls=tc_response.tool_calls,
        )
    )

    # Execute each requested tool and append results.
    for tc_request in tc_response.tool_calls:
        result = registry.execute(tc_request)
        logger.debug(
            "Tool '%s' result (error=%s): %s",
            result.tool_name,
            result.is_error,
            result.content[:80],
        )
        # Append the tool result as a tool-role message.
        conversation.append(
            ChatMessage(
                role=MessageRole.tool,
                content=result.content,
                tool_call_id=result.call_id,
            )
        )

    # --- Round 2: send full conversation including tool results ---
    logger.debug("run_with_tools — sending tool results back (%d messages)", len(conversation))
    final_response: ToolCallingResponse = provider.chat_with_tools(conversation, tool_schemas)

    return _tc_to_llm_response(final_response)


async def arun_with_tools(
    provider: LLMProviderProtocol,
    registry: ToolRegistry,
    messages: list[ChatMessage],
) -> LLMResponse:
    """
    Async variant of ``run_with_tools``.

    Identical logic to the sync version but uses ``achat_with_tools``.
    """
    tool_schemas = registry.tool_schemas()
    logger.debug("arun_with_tools — %d tool(s) available", len(tool_schemas))

    tc_response: ToolCallingResponse = await provider.achat_with_tools(messages, tool_schemas)

    if not tc_response.has_tool_calls:
        logger.debug("arun_with_tools — model responded directly")
        return _tc_to_llm_response(tc_response)

    logger.debug("arun_with_tools — model issued %d tool call(s)", len(tc_response.tool_calls))

    conversation = list(messages)
    conversation.append(
        ChatMessage(
            role=MessageRole.assistant,
            content=tc_response.content or "",
            tool_calls=tc_response.tool_calls,
        )
    )

    for tc_request in tc_response.tool_calls:
        result = registry.execute(tc_request)
        logger.debug(
            "Tool '%s' result (error=%s): %s",
            result.tool_name,
            result.is_error,
            result.content[:80],
        )
        conversation.append(
            ChatMessage(
                role=MessageRole.tool,
                content=result.content,
                tool_call_id=result.call_id,
            )
        )

    final_response: ToolCallingResponse = await provider.achat_with_tools(
        conversation, tool_schemas
    )
    return _tc_to_llm_response(final_response)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tc_to_llm_response(tc: ToolCallingResponse) -> LLMResponse:
    """
    Convert a ``ToolCallingResponse`` to an ``LLMResponse``.

    Used when the model either responded directly or after tool use.
    """
    from rag_assistant.llm.interface import LLMResponse  # avoid circular import

    return LLMResponse(
        content=tc.content or "",
        model=tc.model,
        usage=tc.usage,
        latency_ms=tc.latency_ms,
    )
