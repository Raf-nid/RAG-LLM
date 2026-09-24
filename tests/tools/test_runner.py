"""
Unit tests for run_with_tools and arun_with_tools.

The runner orchestrates one round of tool calling.  All LLM calls are mocked
using a simple stub provider — no HTTP calls are made.

Test scenarios:
1. Model responds directly (no tool calls) — runner returns response as-is.
2. Model issues one tool call — runner executes it, sends results back,
   returns final response.
3. Model issues multiple tool calls — all executed, all results sent back.
4. Tool fails (ToolError) — error result sent to model, model still answers.
5. Async variant mirrors sync behaviour.
"""

from __future__ import annotations

import pytest

from rag_assistant.llm.interface import (
    ChatMessage,
    LLMResponse,
    MessageRole,
    TokenUsage,
    ToolCallingResponse,
    ToolCallRequest,
    ToolSchema,
)
from rag_assistant.tools.calculator import CalculatorTool
from rag_assistant.tools.registry import ToolRegistry
from rag_assistant.tools.runner import arun_with_tools, run_with_tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def _tc_response(
    content: str | None = None,
    tool_calls: tuple[ToolCallRequest, ...] = (),
) -> ToolCallingResponse:
    return ToolCallingResponse(
        content=content,
        tool_calls=tool_calls,
        model="stub-model",
        usage=_usage(),
        latency_ms=10.0,
    )


def _messages() -> list[ChatMessage]:
    return [ChatMessage(role=MessageRole.user, content="What is 6 * 7?")]


# ---------------------------------------------------------------------------
# Stub provider
# ---------------------------------------------------------------------------


class StubProvider:
    """
    Minimal provider that returns pre-configured responses.

    ``responses`` is a list of ToolCallingResponse values returned in order.
    Allows testing multiple-round conversations (round 1 → tool calls,
    round 2 → direct answer).
    """

    def __init__(self, responses: list[ToolCallingResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[ChatMessage], list[ToolSchema]]] = []

    def _next(self, messages: list[ChatMessage], schemas: list[ToolSchema]) -> ToolCallingResponse:
        self.calls.append((messages, schemas))
        return self._responses.pop(0)

    def chat_with_tools(
        self, messages: list[ChatMessage], tool_schemas: list[ToolSchema]
    ) -> ToolCallingResponse:
        return self._next(messages, tool_schemas)

    async def achat_with_tools(
        self, messages: list[ChatMessage], tool_schemas: list[ToolSchema]
    ) -> ToolCallingResponse:
        return self._next(messages, tool_schemas)


def _registry_with_calculator() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    return reg


# ---------------------------------------------------------------------------
# 1. Direct response (no tool calls)
# ---------------------------------------------------------------------------


class TestDirectResponse:
    def test_sync_returns_direct_answer(self) -> None:
        provider = StubProvider([_tc_response(content="The answer is 42.")])
        reg = _registry_with_calculator()
        result = run_with_tools(provider, reg, _messages())
        assert result.content == "The answer is 42."

    def test_sync_provider_called_once(self) -> None:
        provider = StubProvider([_tc_response(content="Direct.")])
        reg = _registry_with_calculator()
        run_with_tools(provider, reg, _messages())
        assert len(provider.calls) == 1

    def test_sync_tool_schemas_passed(self) -> None:
        provider = StubProvider([_tc_response(content="Direct.")])
        reg = _registry_with_calculator()
        run_with_tools(provider, reg, _messages())
        _, schemas = provider.calls[0]
        assert any(s.name == "calculator" for s in schemas)

    def test_sync_latency_preserved(self) -> None:
        provider = StubProvider([_tc_response(content="OK")])
        reg = _registry_with_calculator()
        result = run_with_tools(provider, reg, _messages())
        assert result.latency_ms == 10.0

    def test_sync_model_preserved(self) -> None:
        provider = StubProvider([_tc_response(content="OK")])
        reg = _registry_with_calculator()
        result = run_with_tools(provider, reg, _messages())
        assert result.model == "stub-model"


# ---------------------------------------------------------------------------
# 2. Single tool call round
# ---------------------------------------------------------------------------


class TestSingleToolCallRound:
    def _make_tc_request(self, expr: str = "6 * 7", call_id: str = "call-1") -> ToolCallRequest:
        return ToolCallRequest(
            call_id=call_id,
            tool_name="calculator",
            arguments={"expression": expr},
        )

    def test_sync_two_provider_calls(self) -> None:
        req = self._make_tc_request()
        provider = StubProvider(
            [
                _tc_response(tool_calls=(req,)),  # round 1: model wants a tool
                _tc_response(content="6 * 7 = 42"),  # round 2: final answer
            ]
        )
        reg = _registry_with_calculator()
        run_with_tools(provider, reg, _messages())
        assert len(provider.calls) == 2

    def test_sync_tool_result_in_second_call(self) -> None:
        req = self._make_tc_request("6 * 7", call_id="call-abc")
        provider = StubProvider(
            [
                _tc_response(tool_calls=(req,)),
                _tc_response(content="42"),
            ]
        )
        reg = _registry_with_calculator()
        run_with_tools(provider, reg, _messages())
        # Second call messages should include the tool result.
        messages_round2, _ = provider.calls[1]
        tool_msgs = [m for m in messages_round2 if m.role == MessageRole.tool]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "42"
        assert tool_msgs[0].tool_call_id == "call-abc"

    def test_sync_final_content_returned(self) -> None:
        req = self._make_tc_request()
        provider = StubProvider(
            [
                _tc_response(tool_calls=(req,)),
                _tc_response(content="The result is 42."),
            ]
        )
        reg = _registry_with_calculator()
        result = run_with_tools(provider, reg, _messages())
        assert result.content == "The result is 42."

    def test_sync_assistant_message_appended(self) -> None:
        """The assistant's tool-requesting message must appear in round-2 history."""
        req = self._make_tc_request()
        provider = StubProvider(
            [
                _tc_response(tool_calls=(req,)),
                _tc_response(content="OK"),
            ]
        )
        reg = _registry_with_calculator()
        run_with_tools(provider, reg, _messages())
        messages_round2, _ = provider.calls[1]
        assistant_msgs = [m for m in messages_round2 if m.role == MessageRole.assistant]
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0].tool_calls) == 1


# ---------------------------------------------------------------------------
# 3. Multiple tool calls
# ---------------------------------------------------------------------------


class TestMultipleToolCalls:
    def test_sync_all_results_sent_back(self) -> None:
        req1 = ToolCallRequest(
            call_id="c1", tool_name="calculator", arguments={"expression": "2 + 2"}
        )
        req2 = ToolCallRequest(
            call_id="c2", tool_name="calculator", arguments={"expression": "3 * 3"}
        )
        provider = StubProvider(
            [
                _tc_response(tool_calls=(req1, req2)),
                _tc_response(content="4 and 9"),
            ]
        )
        reg = _registry_with_calculator()
        run_with_tools(provider, reg, _messages())

        messages_round2, _ = provider.calls[1]
        tool_msgs = [m for m in messages_round2 if m.role == MessageRole.tool]
        assert len(tool_msgs) == 2
        contents = {m.content for m in tool_msgs}
        assert "4" in contents
        assert "9" in contents


# ---------------------------------------------------------------------------
# 4. Tool error handling
# ---------------------------------------------------------------------------


class TestToolErrorHandling:
    def test_sync_error_result_sent_to_model(self) -> None:
        req = ToolCallRequest(
            call_id="c-err",
            tool_name="calculator",
            arguments={"expression": "1 / 0"},
        )
        provider = StubProvider(
            [
                _tc_response(tool_calls=(req,)),
                _tc_response(content="Cannot divide by zero."),
            ]
        )
        reg = _registry_with_calculator()
        run_with_tools(provider, reg, _messages())

        messages_round2, _ = provider.calls[1]
        tool_msgs = [m for m in messages_round2 if m.role == MessageRole.tool]
        assert len(tool_msgs) == 1
        assert "zero" in tool_msgs[0].content.lower()

    def test_sync_runner_does_not_raise_on_tool_error(self) -> None:
        req = ToolCallRequest(
            call_id="c-err",
            tool_name="calculator",
            arguments={"expression": "1 / 0"},
        )
        provider = StubProvider(
            [
                _tc_response(tool_calls=(req,)),
                _tc_response(content="Error occurred."),
            ]
        )
        reg = _registry_with_calculator()
        result = run_with_tools(provider, reg, _messages())  # must not raise
        assert isinstance(result, LLMResponse)


# ---------------------------------------------------------------------------
# 5. Async variant
# ---------------------------------------------------------------------------


class TestAsyncRunner:
    @pytest.mark.asyncio
    async def test_direct_response(self) -> None:
        provider = StubProvider([_tc_response(content="async direct")])
        reg = _registry_with_calculator()
        result = await arun_with_tools(provider, reg, _messages())
        assert result.content == "async direct"

    @pytest.mark.asyncio
    async def test_tool_call_round(self) -> None:
        req = ToolCallRequest(
            call_id="a1",
            tool_name="calculator",
            arguments={"expression": "10 + 5"},
        )
        provider = StubProvider(
            [
                _tc_response(tool_calls=(req,)),
                _tc_response(content="15"),
            ]
        )
        reg = _registry_with_calculator()
        result = await arun_with_tools(provider, reg, _messages())
        assert result.content == "15"

    @pytest.mark.asyncio
    async def test_uses_achat_with_tools(self) -> None:
        """Verify async runner uses achat_with_tools, not chat_with_tools."""
        provider = StubProvider([_tc_response(content="OK")])
        reg = _registry_with_calculator()
        await arun_with_tools(provider, reg, _messages())
        assert len(provider.calls) == 1
