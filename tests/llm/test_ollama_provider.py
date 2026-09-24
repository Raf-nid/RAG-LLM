"""
Unit tests for OllamaProvider.

All tests use dependency injection (``client=mock_client``) so no running
Ollama daemon is required.

What is tested
--------------
- Construction succeeds with valid settings.
- Construction raises ``ValueError`` for empty base URL or model name.
- repr() contains model name and base URL, contains no secrets.
- Message role conversion: system, user, assistant (via shared _langchain layer).
- Response parsing: content, model name, token usage, latency.
- Zero-usage fallback when ``usage_metadata`` is None.
- Async path (``achat``) produces the same structure as sync ``chat``.
- ``OllamaProvider`` satisfies the ``LLMProviderProtocol`` structural contract.

What is NOT tested here
-----------------------
- Real HTTP calls to a running Ollama daemon.  Those belong in integration
  tests that explicitly require Ollama to be running.
- Behaviour when the model has not been pulled locally.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from rag_assistant.config import Settings
from rag_assistant.llm.interface import (
    ChatMessage,
    LLMProviderProtocol,
    LLMResponse,
    MessageRole,
)
from rag_assistant.llm.ollama import OllamaProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: object) -> Settings:
    """Build Settings for Ollama unit tests, bypassing .env."""
    base: dict[str, object] = {
        "llm_provider": "ollama",
        "llm_model": "llama3.1",
        "ollama_base_url": "http://localhost:11434",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def _make_ai_message(
    content: str,
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def _make_provider(settings: Settings | None = None) -> tuple[OllamaProvider, MagicMock]:
    """Return an OllamaProvider with an injected mock client."""
    mock_client: MagicMock = MagicMock()
    provider = OllamaProvider(settings or _make_settings(), client=mock_client)
    return provider, mock_client


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestOllamaProviderInit:
    def test_constructs_with_valid_settings(self) -> None:
        provider, _ = _make_provider()
        assert provider._model_name == "llama3.1"

    def test_raises_when_base_url_empty(self) -> None:
        s = _make_settings(ollama_base_url="")
        with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
            OllamaProvider(s)

    def test_raises_when_model_empty(self) -> None:
        s = _make_settings(llm_model="")
        with pytest.raises(ValueError, match="LLM_MODEL"):
            OllamaProvider(s)

    def test_satisfies_provider_protocol(self) -> None:
        """OllamaProvider must satisfy the structural LLMProviderProtocol."""
        provider, _ = _make_provider()
        assert isinstance(provider, LLMProviderProtocol)

    def test_repr_contains_model_and_url(self) -> None:
        provider, _ = _make_provider()
        r = repr(provider)
        assert "llama3.1" in r
        assert "localhost:11434" in r

    def test_custom_base_url_is_stored(self) -> None:
        s = _make_settings(ollama_base_url="http://192.168.1.10:11434")
        provider, _ = _make_provider(s)
        assert provider._base_url == "http://192.168.1.10:11434"


# ---------------------------------------------------------------------------
# Message conversion (exercised end-to-end through the provider)
# ---------------------------------------------------------------------------


class TestMessageConversion:
    def test_user_message_is_sent(self) -> None:
        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("Hi there")

        provider.chat([ChatMessage(role=MessageRole.user, content="Hello")])

        mock_client.invoke.assert_called_once()
        lc_msgs = mock_client.invoke.call_args[0][0]
        assert lc_msgs[0].content == "Hello"

    def test_system_message_type(self) -> None:
        from langchain_core.messages import SystemMessage

        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("ok")

        provider.chat([ChatMessage(role=MessageRole.system, content="Be helpful.")])

        lc_msgs = mock_client.invoke.call_args[0][0]
        assert isinstance(lc_msgs[0], SystemMessage)

    def test_user_message_type(self) -> None:
        from langchain_core.messages import HumanMessage

        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("ok")

        provider.chat([ChatMessage(role=MessageRole.user, content="Question?")])

        lc_msgs = mock_client.invoke.call_args[0][0]
        assert isinstance(lc_msgs[0], HumanMessage)

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
        mock_client.invoke.return_value = _make_ai_message("Local answer.")

        result = provider.chat([ChatMessage(role=MessageRole.user, content="?")])
        assert result.content == "Local answer."

    def test_model_name_from_settings(self) -> None:
        s = _make_settings(llm_model="mistral")
        provider, mock_client = _make_provider(s)
        mock_client.invoke.return_value = _make_ai_message("ok")

        result = provider.chat([ChatMessage(role=MessageRole.user, content="?")])
        assert result.model == "mistral"

    def test_token_usage_is_parsed(self) -> None:
        provider, mock_client = _make_provider()
        mock_client.invoke.return_value = _make_ai_message("ok", input_tokens=8, output_tokens=12)

        result = provider.chat([ChatMessage(role=MessageRole.user, content="?")])
        assert result.usage.prompt_tokens == 8
        assert result.usage.completion_tokens == 12
        assert result.usage.total_tokens == 20

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
            return_value=_make_ai_message("async local answer", input_tokens=6, output_tokens=9)
        )

        result = asyncio.run(
            provider.achat([ChatMessage(role=MessageRole.user, content="async question")])
        )

        assert result.content == "async local answer"
        assert result.usage.prompt_tokens == 6
        assert result.usage.completion_tokens == 9
        assert isinstance(result, LLMResponse)

    def test_achat_calls_ainvoke(self) -> None:
        provider, mock_client = _make_provider()
        mock_client.ainvoke = AsyncMock(return_value=_make_ai_message("ok"))

        asyncio.run(provider.achat([ChatMessage(role=MessageRole.user, content="?")]))

        mock_client.ainvoke.assert_called_once()
