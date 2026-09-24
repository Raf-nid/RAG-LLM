"""
Unit tests for LCEL chain builders and native structured output.

Tests:
1. build_chat_chain     — returns Runnable, processes input, returns str
2. build_structured_chain — returns Runnable, delegates to with_structured_output
3. call_structured_native — calls with_structured_output, returns schema instance
4. acall_structured_native — async variant
5. as_lc_model (Groq)   — returns underlying ChatGroq
6. as_lc_model (Ollama) — returns underlying ChatOllama

All LLM calls are mocked — no HTTP requests are made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from rag_assistant.llm.chain import (
    acall_structured_native,
    build_chat_chain,
    build_structured_chain,
    call_structured_native,
)
from rag_assistant.llm.interface import ChatMessage, MessageRole
from rag_assistant.llm.prompt import build_chat_prompt

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class SimpleOutput(BaseModel):
    answer: str


def _messages() -> list[ChatMessage]:
    return [ChatMessage(role=MessageRole.user, content="What is RAG?")]


def _prompt() -> object:
    return build_chat_prompt("You are a tutor.", "Explain {concept}.")


def _mock_model_for_chain(text_response: str = "test response") -> MagicMock:
    """
    Build a MagicMock that behaves like a BaseChatModel inside LCEL chains.

    Why spec=BaseChatModel?
    -----------------------
    When building a chain with ``prompt | model | parser``, LangChain's ``|``
    operator checks ``isinstance(model, Runnable)``.  A plain ``MagicMock``
    fails this check and gets wrapped in a ``RunnableLambda``, which calls
    the mock as a *callable* (``mock(input)``), not via ``.invoke()``.
    That bypasses our ``return_value`` setup and returns another ``MagicMock``.

    Speccing against ``BaseChatModel`` (itself a ``Runnable`` subclass) makes
    ``isinstance(mock, Runnable)`` return True.  LangChain then calls
    ``mock.invoke(messages, config)`` correctly, returning our ``AIMessage``.
    """
    from langchain_core.language_models import BaseChatModel

    mock = MagicMock(spec=BaseChatModel)
    mock.invoke.return_value = AIMessage(content=text_response)
    return mock


# ---------------------------------------------------------------------------
# 1. build_chat_chain
# ---------------------------------------------------------------------------


class TestBuildChatChain:
    def test_returns_runnable(self) -> None:
        prompt = _prompt()
        model = _mock_model_for_chain()
        chain = build_chat_chain(prompt, model)  # type: ignore[arg-type]
        assert isinstance(chain, Runnable)

    def test_chain_is_composable(self) -> None:
        """Verify | operator works on the result (i.e. it is a real Runnable)."""
        prompt = _prompt()
        model = _mock_model_for_chain()
        chain = build_chat_chain(prompt, model)  # type: ignore[arg-type]
        # A Runnable must expose invoke/ainvoke.
        assert hasattr(chain, "invoke")
        assert hasattr(chain, "ainvoke")

    def test_chain_output_type(self) -> None:
        """build_chat_chain ends with StrOutputParser so output must be str."""
        prompt = build_chat_prompt("S.", "Hello {name}.")
        model = _mock_model_for_chain("Hello World")
        chain = build_chat_chain(prompt, model)  # type: ignore[arg-type]
        result = chain.invoke({"name": "World"})
        assert isinstance(result, str)

    def test_chain_returns_model_content(self) -> None:
        prompt = build_chat_prompt("S.", "Question: {q}.")
        model = _mock_model_for_chain("Embeddings encode meaning.")
        chain = build_chat_chain(prompt, model)  # type: ignore[arg-type]
        result = chain.invoke({"q": "What are embeddings?"})
        assert result == "Embeddings encode meaning."


# ---------------------------------------------------------------------------
# 2. build_structured_chain
# ---------------------------------------------------------------------------


class TestBuildStructuredChain:
    def test_returns_runnable(self) -> None:
        prompt = _prompt()
        model = MagicMock()
        mock_structured = MagicMock(spec=Runnable)
        mock_structured.invoke.return_value = SimpleOutput(answer="test")
        model.with_structured_output.return_value = mock_structured
        chain = build_structured_chain(prompt, model, SimpleOutput)  # type: ignore[arg-type]
        assert isinstance(chain, Runnable)

    def test_calls_with_structured_output(self) -> None:
        """build_structured_chain must call model.with_structured_output(schema)."""
        prompt = _prompt()
        model = MagicMock()
        mock_structured = MagicMock(spec=Runnable)
        mock_structured.invoke.return_value = SimpleOutput(answer="ok")
        model.with_structured_output.return_value = mock_structured
        build_structured_chain(prompt, model, SimpleOutput)  # type: ignore[arg-type]
        model.with_structured_output.assert_called_once_with(SimpleOutput)

    def test_chain_returns_schema_instance(self) -> None:
        prompt = build_chat_prompt("S.", "Answer: {q}.")
        model = MagicMock()
        expected = SimpleOutput(answer="42")
        mock_structured = MagicMock(spec=Runnable)
        mock_structured.invoke.return_value = expected
        model.with_structured_output.return_value = mock_structured
        chain = build_structured_chain(prompt, model, SimpleOutput)  # type: ignore[arg-type]
        result = chain.invoke({"q": "six times seven"})
        assert isinstance(result, SimpleOutput)
        assert result.answer == "42"


# ---------------------------------------------------------------------------
# 3. call_structured_native
# ---------------------------------------------------------------------------


class TestCallStructuredNative:
    def _make_model(self, return_value: BaseModel) -> MagicMock:
        model = MagicMock()
        structured = MagicMock()
        structured.invoke.return_value = return_value
        model.with_structured_output.return_value = structured
        return model

    def test_returns_schema_instance(self) -> None:
        expected = SimpleOutput(answer="42")
        model = self._make_model(expected)
        result = call_structured_native(model, _messages(), SimpleOutput)
        assert isinstance(result, SimpleOutput)
        assert result.answer == "42"

    def test_calls_with_structured_output(self) -> None:
        expected = SimpleOutput(answer="ok")
        model = self._make_model(expected)
        call_structured_native(model, _messages(), SimpleOutput)
        model.with_structured_output.assert_called_once_with(SimpleOutput)

    def test_invoke_is_called_once(self) -> None:
        expected = SimpleOutput(answer="ok")
        model = self._make_model(expected)
        call_structured_native(model, _messages(), SimpleOutput)
        model.with_structured_output.return_value.invoke.assert_called_once()

    def test_messages_converted_to_lc_format(self) -> None:
        """Verify to_langchain_messages is called (messages reach the model)."""
        expected = SimpleOutput(answer="ok")
        model = self._make_model(expected)
        messages = [
            ChatMessage(role=MessageRole.system, content="System."),
            ChatMessage(role=MessageRole.user, content="Question?"),
        ]
        call_structured_native(model, messages, SimpleOutput)
        call_args = model.with_structured_output.return_value.invoke.call_args
        lc_msgs = call_args[0][0]
        assert len(lc_msgs) == 2

    def test_type_error_raised_on_wrong_return_type(self) -> None:
        model = MagicMock()
        structured = MagicMock()
        # Simulate model returning wrong type
        structured.invoke.return_value = {"answer": "42"}  # dict, not SimpleOutput
        model.with_structured_output.return_value = structured
        with pytest.raises(TypeError, match="Expected SimpleOutput"):
            call_structured_native(model, _messages(), SimpleOutput)


# ---------------------------------------------------------------------------
# 4. acall_structured_native
# ---------------------------------------------------------------------------


class TestACallStructuredNative:
    def _make_async_model(self, return_value: BaseModel) -> MagicMock:
        model = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value=return_value)
        model.with_structured_output.return_value = structured
        return model

    @pytest.mark.asyncio
    async def test_returns_schema_instance(self) -> None:
        expected = SimpleOutput(answer="async answer")
        model = self._make_async_model(expected)
        result = await acall_structured_native(model, _messages(), SimpleOutput)
        assert isinstance(result, SimpleOutput)
        assert result.answer == "async answer"

    @pytest.mark.asyncio
    async def test_uses_ainvoke_not_invoke(self) -> None:
        expected = SimpleOutput(answer="ok")
        model = self._make_async_model(expected)
        await acall_structured_native(model, _messages(), SimpleOutput)
        model.with_structured_output.return_value.ainvoke.assert_awaited_once()
        model.with_structured_output.return_value.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_type_error_raised_on_wrong_return_type(self) -> None:
        model = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value={"bad": "type"})
        model.with_structured_output.return_value = structured
        with pytest.raises(TypeError):
            await acall_structured_native(model, _messages(), SimpleOutput)


# ---------------------------------------------------------------------------
# 5 & 6. as_lc_model on providers
# ---------------------------------------------------------------------------


def _groq_settings() -> object:
    from rag_assistant.config import Settings

    return Settings(
        _env_file=None,
        llm_provider="groq",
        llm_model="qwen/qwen3.8-27b",
        groq_api_key="sk-test-key-not-real",
    )


def _ollama_settings() -> object:
    from rag_assistant.config import Settings

    return Settings(
        _env_file=None,
        llm_provider="ollama",
        llm_model="llama3.1",
        ollama_base_url="http://localhost:11434",
    )


class TestAsLcModel:
    def test_groq_as_lc_model_returns_chat_groq(self) -> None:
        from langchain_groq import ChatGroq

        from rag_assistant.llm.groq import GroqProvider

        with patch("rag_assistant.llm.groq.ChatGroq") as mock_chat_groq_cls:
            mock_instance = MagicMock(spec=ChatGroq)
            mock_chat_groq_cls.return_value = mock_instance
            provider = GroqProvider(_groq_settings())  # type: ignore[arg-type]
            lc_model = provider.as_lc_model()
            assert lc_model is mock_instance

    def test_ollama_as_lc_model_returns_chat_ollama(self) -> None:
        from langchain_ollama import ChatOllama

        from rag_assistant.llm.ollama import OllamaProvider

        with patch("rag_assistant.llm.ollama.ChatOllama") as mock_chat_ollama_cls:
            mock_instance = MagicMock(spec=ChatOllama)
            mock_chat_ollama_cls.return_value = mock_instance
            provider = OllamaProvider(_ollama_settings())  # type: ignore[arg-type]
            lc_model = provider.as_lc_model()
            assert lc_model is mock_instance

    def test_groq_as_lc_model_same_instance_every_call(self) -> None:
        """as_lc_model must return the same object, not create new clients."""
        from rag_assistant.llm.groq import GroqProvider

        with patch("rag_assistant.llm.groq.ChatGroq"):
            provider = GroqProvider(_groq_settings())  # type: ignore[arg-type]
            assert provider.as_lc_model() is provider.as_lc_model()

    def test_ollama_as_lc_model_same_instance_every_call(self) -> None:
        from rag_assistant.llm.ollama import OllamaProvider

        with patch("rag_assistant.llm.ollama.ChatOllama"):
            provider = OllamaProvider(_ollama_settings())  # type: ignore[arg-type]
            assert provider.as_lc_model() is provider.as_lc_model()
