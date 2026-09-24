"""
Unit tests for the LLM provider factory.

The factory is the single place that maps configuration to a concrete
provider class.  Tests here verify the routing logic without caring about
what the provider actually does.
"""

from unittest.mock import patch

import pytest

from rag_assistant.config import Settings
from rag_assistant.llm.factory import create_llm_provider
from rag_assistant.llm.groq import GroqProvider
from rag_assistant.llm.interface import LLMProviderProtocol


def _make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "llm_provider": "groq",
        "llm_model": "openai/gpt-oss-20b",
        "groq_api_key": "sk-test-key",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class TestCreateLLMProvider:
    def test_groq_provider_selected_for_groq_config(self) -> None:
        """Factory returns a GroqProvider when llm_provider=groq."""
        s = _make_settings(llm_provider="groq")
        # Patch ChatGroq so no real HTTP client is initialised
        with patch("rag_assistant.llm.groq.ChatGroq"):
            provider = create_llm_provider(s)
        assert isinstance(provider, GroqProvider)

    def test_returned_provider_satisfies_protocol(self) -> None:
        """Whatever the factory returns must satisfy LLMProviderProtocol."""
        s = _make_settings()
        with patch("rag_assistant.llm.groq.ChatGroq"):
            provider = create_llm_provider(s)
        assert isinstance(provider, LLMProviderProtocol)

    def test_unimplemented_provider_raises(self) -> None:
        """Unimplemented providers must raise NotImplementedError, not silently fail."""
        s = _make_settings(llm_provider="openai", openai_api_key="sk-test")
        with pytest.raises(NotImplementedError, match="openai"):
            create_llm_provider(s)
