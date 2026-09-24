"""
Unit tests for GroqProvider.

All tests use dependency injection (``client=mock_client``) so no real
HTTP calls are made and no real API key is required.

What is tested
--------------
- Constructor rejects missing API key immediately (fail-fast).
- repr() never exposes the API key.
- Message role conversion: system, user, assistant.
- Response parsing: content, model name, token usage, latency.
- Zero-usage fallback when ``usage_metadata`` is None.
- Async path (``achat``) produces the same structure as sync ``chat``.
- ``GroqProvider`` satisfies the ``LLMProviderProtocol`` structural contract.

What is NOT tested here
-----------------------
- Real HTTP calls to Groq.  Those belong in integration tests that require
  a live API key and are explicitly skipped in CI unless the key is present.
- Retry / rate-limit behaviour.  That is a service-layer concern.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from rag_assistant.config import Settings
from rag_assistant.llm.groq import GroqProvider
from rag_assistant.llm.interface import (
    ChatMessage,
    LLMProviderProtocol,
    LLMResponse,
    MessageRole,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: object) -> Settings:
    """Build a Settings object suitable for unit tests."""
    base: dict[str, object] = {
        "llm_provider": "groq",
        "llm_model": "openai/gpt-oss-20b",
        "groq_api_key": "sk-test-key-not-real",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _make_ai_message(
    content: str,
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> AIMessage:
    """Build a fake AIMessage as LangChain would return it."""
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def _make_provider(settings: Settings | None = None) -> tuple[GroqProvider, MagicMock]:
    """Return a GroqProvider with an injected mock client."""
    mock_client: MagicMock = MagicMock()
    provider = GroqProvider(settings or _make_settings(), client=mock_client)
    return provider, mock_client


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestGroqProviderInit:
    def test_raises_when_api_key_missing(self) -> None:
        """Provider must fail fast if no API key is configured."""
        s = _make_settings(groq_api_key=None)
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            GroqProvider(s)

    def test_repr_does_not_contain_api_key(self) -> None:
        """repr() must never expose secret values."""
        provider, _ = _make_provider()
        assert "sk-test-key-not-real" not in repr(provider)

    def test_repr_contains_model_name(self) -> None:
        s = _make_settings(llm_model="mixtral-8x7b-32768")
        provider, _ = _make_provider(s)
        assert "mixtral-8x7b-32768" in repr(provider)

    def test_satisfies_provider_protocol(self) -> None:
        """GroqProvider must satisfy the structural LLMProviderProtocol."""
        provider, _ = _make_provider()
        assert isinstance(provider, LLMProviderProtocol)


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestMessageConversion:
    """Verify that our domain ChatMessage types map to the right LangChain types."""

    def test_user_message_is_sent(self) -> None:
        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("Hi there")

        messages = [ChatMessage(role=MessageRole.user, content="Hello")]
        provider.chat(messages)

        mock_client.invoke.assert_called_once()
        lc_msgs = mock_client.invoke.call_args[0][0]
        assert len(lc_msgs) == 1
        assert lc_msgs[0].content == "Hello"

    def test_system_message_type(self) -> None:
        from langchain_core.messages import SystemMessage

        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("ok")

        provider.chat([ChatMessage(role=MessageRole.system, content="You are helpful.")])

        lc_msgs = mock_client.invoke.call_args[0][0]
        assert isinstance(lc_msgs[0], SystemMessage)

    def test_user_message_type(self) -> None:
        from langchain_core.messages import HumanMessage

        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("ok")

        provider.chat([ChatMessage(role=MessageRole.user, content="Question?")])

        lc_msgs = mock_client.invoke.call_args[0][0]
        assert isinstance(lc_msgs[0], HumanMessage)

    def test_assistant_message_type(self) -> None:
        from langchain_core.messages import AIMessage as LCAIMessage

        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("ok")

        provider.chat([ChatMessage(role=MessageRole.assistant, content="Previous answer.")])

        lc_msgs = mock_client.invoke.call_args[0][0]
        assert isinstance(lc_msgs[0], LCAIMessage)

    def test_multi_turn_conversation_order(self) -> None:
        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("ok")

        messages = [
            ChatMessage(role=MessageRole.system, content="You are an expert."),
            ChatMessage(role=MessageRole.user, content="What is RAG?"),
            ChatMessage(role=MessageRole.assistant, content="RAG stands for..."),
            ChatMessage(role=MessageRole.user, content="Tell me more."),
        ]
        provider.chat(messages)

        lc_msgs = mock_client.invoke.call_args[0][0]
        assert len(lc_msgs) == 4
        assert lc_msgs[0].content == "You are an expert."
        assert lc_msgs[-1].content == "Tell me more."


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_content_is_preserved(self) -> None:
        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("The answer is 42.")

        result = provider.chat([ChatMessage(role=MessageRole.user, content="?")])
        assert result.content == "The answer is 42."

    def test_model_name_from_settings(self) -> None:
        s = _make_settings(llm_model="gemma2-9b-it")
        provider, mock_client = _make_provider(s)
        mock_client.invoke.return_value = _make_ai_message("ok")

        result = provider.chat([ChatMessage(role=MessageRole.user, content="?")])
        assert result.model == "gemma2-9b-it"

    def test_token_usage_is_parsed(self) -> None:
        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("ok", input_tokens=15, output_tokens=7)

        result = provider.chat([ChatMessage(role=MessageRole.user, content="?")])
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 7
        assert result.usage.total_tokens == 22

    def test_latency_is_non_negative(self) -> None:
        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("ok")

        result = provider.chat([ChatMessage(role=MessageRole.user, content="?")])
        assert result.latency_ms >= 0.0

    def test_response_is_llm_response_type(self) -> None:
        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("ok")

        result = provider.chat([ChatMessage(role=MessageRole.user, content="?")])
        assert isinstance(result, LLMResponse)

    def test_usage_metadata_none_returns_zeros(self) -> None:
        """Provider must not crash when the model omits usage data."""
        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = AIMessage(content="ok", usage_metadata=None)

        result = provider.chat([ChatMessage(role=MessageRole.user, content="?")])
        assert result.usage.total_tokens == 0


# ---------------------------------------------------------------------------
# Async path
# ---------------------------------------------------------------------------


class TestAsyncChat:
    def test_achat_returns_llm_response(self) -> None:
        provider, mock_client = _make_provider()
        mock_client.ainvoke = AsyncMock(
            return_value=_make_ai_message("async answer", input_tokens=5, output_tokens=8)
        )

        result = asyncio.run(
            provider.achat([ChatMessage(role=MessageRole.user, content="async question")])
        )

        assert result.content == "async answer"
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 8
        assert isinstance(result, LLMResponse)

    def test_achat_calls_ainvoke(self) -> None:
        provider, mock_client = _make_provider()
        mock_client.ainvoke = AsyncMock(return_value=_make_ai_message("ok"))

        asyncio.run(provider.achat([ChatMessage(role=MessageRole.user, content="?")]))

        mock_client.ainvoke.assert_called_once()
