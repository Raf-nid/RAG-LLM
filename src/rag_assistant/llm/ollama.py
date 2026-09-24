"""
Ollama LLM provider.

Wraps LangChain's ``ChatOllama`` behind ``LLMProviderProtocol``.
Nothing outside this module should import ``ChatOllama`` or any LangChain type.

Ollama differences from Groq
-----------------------------
- No API key required — Ollama runs locally.
- Requires a running ``ollama serve`` process at ``OLLAMA_BASE_URL``.
- Model names refer to locally-pulled models (e.g. ``llama3.1``, ``mistral``).
- No rate limits or usage costs.

Tool calling with Ollama
------------------------
Tool calling support depends on the model.  Models known to work well:
``llama3.1``, ``qwen2.5``, ``mistral-nemo``.  Smaller quantised models (3B, 7B)
may produce malformed tool call requests.  The provider implementation is
identical to Groq — differences are handled at the model level, not in code.

Failure modes
-------------
- ``httpx.ConnectError`` — Ollama daemon not running at configured URL.
- ``ollama.ResponseError`` — model not found locally (run ``ollama pull <model>``).
"""

import logging
import time

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from rag_assistant.config import Settings

from ._langchain import (
    parse_tool_calls,
    parse_usage,
    to_langchain_messages,
    tool_schema_to_lc_dict,
)
from .interface import (
    ChatMessage,
    LLMResponse,
    ToolCallingResponse,
    ToolSchema,
)

logger = logging.getLogger(__name__)

__all__ = ["OllamaProvider"]


class OllamaProvider:
    """
    LLM provider backed by a local Ollama instance.

    Parameters
    ----------
    settings:
        Application settings.  Uses ``llm_model`` and ``ollama_base_url``.
    client:
        Optional pre-built ``ChatOllama`` for dependency injection in tests.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: ChatOllama | None = None,
    ) -> None:
        if not settings.ollama_base_url:
            raise ValueError("OLLAMA_BASE_URL must not be empty when LLM_PROVIDER=ollama.")
        if not settings.llm_model:
            raise ValueError(
                "LLM_MODEL must not be empty when LLM_PROVIDER=ollama. "
                "Set it to a model you have pulled locally (e.g. 'llama3.1')."
            )

        self._model_name: str = settings.llm_model
        self._base_url: str = settings.ollama_base_url
        self._client: ChatOllama = client or ChatOllama(
            model=self._model_name,
            base_url=self._base_url,
        )
        logger.debug(
            "OllamaProvider initialised (model=%s, base_url=%s)", self._model_name, self._base_url
        )

    # ------------------------------------------------------------------
    # Standard chat
    # ------------------------------------------------------------------

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Synchronous chat completion."""
        lc_messages = to_langchain_messages(messages)
        logger.debug("OllamaProvider.chat — %d message(s)", len(messages))

        t0 = time.perf_counter()
        response: AIMessage = self._client.invoke(lc_messages)  # type: ignore[assignment]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = LLMResponse(
            content=str(response.content),
            model=self._model_name,
            usage=parse_usage(response, provider_name="ollama"),
            latency_ms=round(latency_ms, 2),
        )
        logger.debug(
            "OllamaProvider.chat — done (%.0fms, %d tokens)",
            result.latency_ms,
            result.usage.total_tokens,
        )
        return result

    async def achat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Asynchronous chat completion."""
        lc_messages = to_langchain_messages(messages)
        logger.debug("OllamaProvider.achat — %d message(s)", len(messages))

        t0 = time.perf_counter()
        response: AIMessage = await self._client.ainvoke(lc_messages)  # type: ignore[assignment]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = LLMResponse(
            content=str(response.content),
            model=self._model_name,
            usage=parse_usage(response, provider_name="ollama"),
            latency_ms=round(latency_ms, 2),
        )
        logger.debug(
            "OllamaProvider.achat — done (%.0fms, %d tokens)",
            result.latency_ms,
            result.usage.total_tokens,
        )
        return result

    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        messages: list[ChatMessage],
        tool_schemas: list[ToolSchema],
    ) -> ToolCallingResponse:
        """
        Synchronous chat with tool definitions bound.

        Ollama uses the same API as Groq for tool calling.  Reliability
        varies by model — prefer ``llama3.1`` or ``qwen2.5`` for tools.
        """
        lc_messages = to_langchain_messages(messages)
        lc_tools = [tool_schema_to_lc_dict(s) for s in tool_schemas]
        logger.debug("OllamaProvider.chat_with_tools — %d tool(s)", len(lc_tools))

        t0 = time.perf_counter()
        response: AIMessage = self._client.bind_tools(lc_tools).invoke(lc_messages)  # type: ignore[assignment]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        tool_calls = parse_tool_calls(response)
        result = ToolCallingResponse(
            content=str(response.content) if response.content else None,
            tool_calls=tool_calls,
            model=self._model_name,
            usage=parse_usage(response, provider_name="ollama"),
            latency_ms=round(latency_ms, 2),
        )
        logger.debug(
            "OllamaProvider.chat_with_tools — done (%.0fms, tool_calls=%d)",
            result.latency_ms,
            len(result.tool_calls),
        )
        return result

    async def achat_with_tools(
        self,
        messages: list[ChatMessage],
        tool_schemas: list[ToolSchema],
    ) -> ToolCallingResponse:
        """Asynchronous chat with tool definitions bound."""
        lc_messages = to_langchain_messages(messages)
        lc_tools = [tool_schema_to_lc_dict(s) for s in tool_schemas]
        logger.debug("OllamaProvider.achat_with_tools — %d tool(s)", len(lc_tools))

        t0 = time.perf_counter()
        response: AIMessage = await self._client.bind_tools(lc_tools).ainvoke(lc_messages)  # type: ignore[assignment]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        tool_calls = parse_tool_calls(response)
        result = ToolCallingResponse(
            content=str(response.content) if response.content else None,
            tool_calls=tool_calls,
            model=self._model_name,
            usage=parse_usage(response, provider_name="ollama"),
            latency_ms=round(latency_ms, 2),
        )
        logger.debug(
            "OllamaProvider.achat_with_tools — done (%.0fms, tool_calls=%d)",
            result.latency_ms,
            len(result.tool_calls),
        )
        return result

    def as_lc_model(self) -> ChatOllama:
        """
        Return the underlying ``ChatOllama`` instance.

        Use this when you need to compose LCEL chains or call
        ``model.with_structured_output(schema)`` directly.

        Note: reliability of ``with_structured_output`` varies by model.
        Prefer ``llama3.1`` or ``qwen2.5:7b+`` for structured outputs.
        """
        return self._client

    def __repr__(self) -> str:
        return f"OllamaProvider(model={self._model_name!r}, base_url={self._base_url!r})"
