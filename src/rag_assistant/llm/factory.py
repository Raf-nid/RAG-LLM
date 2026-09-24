"""
LLM provider factory.

This is the only place in the codebase that decides which concrete provider
class to instantiate based on the configuration.  Application code calls
``create_llm_provider(settings)`` and receives an ``LLMProviderProtocol``
— it never names a specific provider class.

Adding a new provider (e.g. Ollama) means:
  1. Create ``src/rag_assistant/llm/ollama.py`` with an ``OllamaProvider`` class.
  2. Add a branch to the ``if`` chain below.
  3. No other file changes required.
"""

from rag_assistant.config import LLMProvider, Settings

from .groq import GroqProvider
from .interface import LLMProviderProtocol


def create_llm_provider(settings: Settings) -> LLMProviderProtocol:
    """
    Instantiate and return the configured LLM provider.

    Parameters
    ----------
    settings:
        Application settings.  The ``llm_provider`` field determines which
        concrete implementation is returned.

    Returns
    -------
    LLMProviderProtocol
        A ready-to-use provider.  The caller does not need to know the
        concrete type.

    Raises
    ------
    ValueError
        If the configured provider is not yet implemented.
    """
    if settings.llm_provider == LLMProvider.groq:
        return GroqProvider(settings)

    raise NotImplementedError(
        f"LLM provider {settings.llm_provider!r} is not yet implemented. "
        f"Currently supported: {[p.value for p in LLMProvider]}"
    )
