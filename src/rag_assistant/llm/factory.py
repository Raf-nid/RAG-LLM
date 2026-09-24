"""
LLM provider factory.

This is the only place in the codebase that maps a provider name to a
concrete implementation class.  Application code calls
``create_llm_provider(settings)`` and receives an ``LLMProviderProtocol``;
it never names a specific provider class.

Adding a new provider means:
  1. Create ``src/rag_assistant/llm/<name>.py`` with a ``<Name>Provider`` class
     that satisfies ``LLMProviderProtocol``.
  2. Add ``LLMProvider.<name>`` to ``config.py``.
  3. Add one branch to the ``if`` chain below.
  4. No other file changes required.
"""

from rag_assistant.config import LLMProvider, Settings

from .groq import GroqProvider
from .interface import LLMProviderProtocol
from .ollama import OllamaProvider


def create_llm_provider(settings: Settings) -> LLMProviderProtocol:
    """
    Instantiate and return the configured LLM provider.

    The caller receives an ``LLMProviderProtocol`` and does not need to
    know which concrete class was returned.

    Parameters
    ----------
    settings:
        Application settings.  ``llm_provider`` determines which
        implementation is returned; ``llm_model`` and provider-specific
        fields (e.g. ``groq_api_key``, ``ollama_base_url``) are forwarded
        to the provider constructor.

    Raises
    ------
    ValueError
        If provider-specific required configuration is missing
        (e.g. ``GROQ_API_KEY`` for Groq).
    NotImplementedError
        If ``llm_provider`` names a provider that is not yet implemented.
        This should only happen if ``LLMProvider`` enum is extended without
        adding a matching branch here.
    """
    if settings.llm_provider == LLMProvider.groq:
        return GroqProvider(settings)

    if settings.llm_provider == LLMProvider.ollama:
        return OllamaProvider(settings)

    # This branch is unreachable under normal operation because pydantic
    # validates llm_provider against the LLMProvider enum at startup.
    # It is kept as a safety net for future enum additions.
    raise NotImplementedError(  # pragma: no cover
        f"LLM provider {settings.llm_provider!r} is not implemented. "
        f"Supported providers: {[p.value for p in LLMProvider]}"
    )
