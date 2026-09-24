"""
Unit tests for the LLM provider factory.

The factory is the single place that maps ``settings.llm_provider`` to a
concrete provider class.  Tests verify the routing logic, not what the
provider itself does.

Coverage goals
--------------
- ``llm_provider=groq``   → ``GroqProvider`` instance
- ``llm_provider=ollama`` → ``OllamaProvider`` instance
- Every returned provider satisfies ``LLMProviderProtocol``
- The factory never constructs a real HTTP client (patched away)
"""

from unittest.mock import patch

from rag_assistant.config import Settings
from rag_assistant.llm.factory import create_llm_provider
from rag_assistant.llm.groq import GroqProvider
from rag_assistant.llm.interface import LLMProviderProtocol
from rag_assistant.llm.ollama import OllamaProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _groq_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "llm_provider": "groq",
        "llm_model": "qwen/qwen3.8-27b",
        "groq_api_key": "sk-test-key",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def _ollama_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "llm_provider": "ollama",
        "llm_model": "llama3.1",
        "ollama_base_url": "http://localhost:11434",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------


class TestProviderRouting:
    def test_groq_config_returns_groq_provider(self) -> None:
        """Factory returns a GroqProvider when llm_provider=groq."""
        with patch("rag_assistant.llm.groq.ChatGroq"):
            provider = create_llm_provider(_groq_settings())
        assert isinstance(provider, GroqProvider)

    def test_ollama_config_returns_ollama_provider(self) -> None:
        """Factory returns an OllamaProvider when llm_provider=ollama."""
        with patch("rag_assistant.llm.ollama.ChatOllama"):
            provider = create_llm_provider(_ollama_settings())
        assert isinstance(provider, OllamaProvider)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_groq_provider_satisfies_protocol(self) -> None:
        with patch("rag_assistant.llm.groq.ChatGroq"):
            provider = create_llm_provider(_groq_settings())
        assert isinstance(provider, LLMProviderProtocol)

    def test_ollama_provider_satisfies_protocol(self) -> None:
        with patch("rag_assistant.llm.ollama.ChatOllama"):
            provider = create_llm_provider(_ollama_settings())
        assert isinstance(provider, LLMProviderProtocol)


# ---------------------------------------------------------------------------
# Configuration forwarding
# ---------------------------------------------------------------------------


class TestConfigurationForwarding:
    def test_groq_uses_model_from_settings(self) -> None:
        with patch("rag_assistant.llm.groq.ChatGroq"):
            provider = create_llm_provider(_groq_settings(llm_model="allam-2-7b"))
        assert isinstance(provider, GroqProvider)
        assert provider._model_name == "allam-2-7b"

    def test_ollama_uses_model_from_settings(self) -> None:
        with patch("rag_assistant.llm.ollama.ChatOllama"):
            provider = create_llm_provider(_ollama_settings(llm_model="mistral"))
        assert isinstance(provider, OllamaProvider)
        assert provider._model_name == "mistral"

    def test_ollama_uses_custom_base_url(self) -> None:
        with patch("rag_assistant.llm.ollama.ChatOllama"):
            provider = create_llm_provider(
                _ollama_settings(ollama_base_url="http://gpu-server:11434")
            )
        assert isinstance(provider, OllamaProvider)
        assert provider._base_url == "http://gpu-server:11434"
