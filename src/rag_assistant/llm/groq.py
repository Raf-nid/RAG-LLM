"""
Groq LLM provider.

Wraps LangChain's ``ChatGroq`` inside our application-level interface.
Nothing outside this module should import ``ChatGroq`` or any other
LangChain type.

Failure modes to be aware of
-----------------------------
- ``groq.APIConnectionError``   — network failure or DNS issue
- ``groq.AuthenticationError``  — invalid or expired API key
- ``groq.RateLimitError``       — token or request quota exceeded
- ``groq.APIStatusError``       — unexpected HTTP status from Groq
- ``ValidationError`` on init   — missing API key caught before any HTTP call

These exceptions propagate to the caller.  Retry and fallback logic
belongs at the service layer, not here.
"""

import logging
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from rag_assistant.config import Settings

from .interface import ChatMessage, LLMResponse, MessageRole, TokenUsage

logger = logging.getLogger(__name__)


def _to_langchain_messages(messages: list[ChatMessage]) -> list[BaseMessage]:
    """Convert our domain messages to LangChain message objects."""
    mapping: dict[MessageRole, type[BaseMessage]] = {
        MessageRole.system: SystemMessage,
        MessageRole.user: HumanMessage,
        MessageRole.assistant: AIMessage,
    }
    result: list[BaseMessage] = []
    for msg in messages:
        cls = mapping.get(msg.role)
        if cls is None:
            raise ValueError(f"Unsupported message role: {msg.role!r}")
        result.append(cls(content=msg.content))
    return result


def _parse_usage(response: AIMessage) -> TokenUsage:
    """Extract token counts from the LangChain response."""
    meta = response.usage_metadata
    if meta is None:
        # Some models / streaming modes do not return usage.
        # Return zeros rather than crashing; the caller can decide whether
        # zero usage is acceptable.
        logger.warning("usage_metadata was None in Groq response; token counts unavailable")
        return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    return TokenUsage(
        prompt_tokens=meta["input_tokens"],
        completion_tokens=meta["output_tokens"],
        total_tokens=meta["total_tokens"],
    )


class GroqProvider:
    """
    LLM provider backed by the Groq API via LangChain.

    Construction validates the API key immediately so that the error
    surfaces at startup rather than at the first request.

    The ``client`` parameter exists for dependency injection in tests:
    pass a mock ``ChatGroq``-compatible object to avoid real HTTP calls.

    Parameters
    ----------
    settings:
        Application settings.  Must have ``groq_api_key`` set when
        ``llm_provider == "groq"``.
    client:
        Optional pre-built ``ChatGroq`` instance.  If ``None``, one is
        created from ``settings``.  Use only in tests.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: ChatGroq | None = None,
    ) -> None:
        if settings.groq_api_key is None:
            raise ValueError(
                "GROQ_API_KEY must be set when LLM_PROVIDER=groq. "
                "Add it to your .env file or set the environment variable."
            )

        self._model_name: str = settings.llm_model
        self._client: ChatGroq = client or ChatGroq(
            model_name=self._model_name,
            groq_api_key=settings.groq_api_key.get_secret_value(),  # type: ignore[arg-type]
        )
        logger.debug("GroqProvider initialised (model=%s)", self._model_name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Synchronous chat completion."""
        lc_messages = _to_langchain_messages(messages)
        logger.debug("GroqProvider.chat — sending %d message(s)", len(messages))

        t0 = time.perf_counter()
        response: AIMessage = self._client.invoke(lc_messages)  # type: ignore[assignment]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = LLMResponse(
            content=str(response.content),
            model=self._model_name,
            usage=_parse_usage(response),
            latency_ms=round(latency_ms, 2),
        )
        logger.debug(
            "GroqProvider.chat — done (latency=%.0fms tokens=%d)",
            result.latency_ms,
            result.usage.total_tokens,
        )
        return result

    async def achat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Asynchronous chat completion."""
        lc_messages = _to_langchain_messages(messages)
        logger.debug("GroqProvider.achat — sending %d message(s)", len(messages))

        t0 = time.perf_counter()
        response: AIMessage = await self._client.ainvoke(lc_messages)  # type: ignore[assignment]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = LLMResponse(
            content=str(response.content),
            model=self._model_name,
            usage=_parse_usage(response),
            latency_ms=round(latency_ms, 2),
        )
        logger.debug(
            "GroqProvider.achat — done (latency=%.0fms tokens=%d)",
            result.latency_ms,
            result.usage.total_tokens,
        )
        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        # Deliberately never includes the API key or any secret.
        return f"GroqProvider(model={self._model_name!r})"
