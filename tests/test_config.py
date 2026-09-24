"""
Tests for application configuration (Settings).

These tests verify that:
  - Settings loads with all defaults without requiring a .env file.
  - Enum fields reject invalid values.
  - Secret fields are not accidentally exposed as plain strings.
  - Provider fields can be overridden via environment variables.
"""

import pytest
from pydantic import ValidationError

from rag_assistant.config import Environment, LLMProvider, Settings


def _settings(**overrides: object) -> Settings:
    """Build Settings without reading the local .env file."""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class TestSettingsDefaults:
    """Settings should load safely with no environment variables set."""

    def test_default_environment(self) -> None:
        s = _settings()
        assert s.environment == Environment.development

    def test_default_llm_provider(self) -> None:
        s = _settings()
        assert s.llm_provider == LLMProvider.groq

    def test_default_llm_model(self) -> None:
        s = _settings()
        assert s.llm_model == "qwen/qwen3.8-27b"

    def test_no_api_keys_by_default(self) -> None:
        s = _settings()
        assert s.groq_api_key is None
        assert s.langsmith_api_key is None

    def test_langchain_tracing_disabled_by_default(self) -> None:
        s = _settings()
        assert s.langchain_tracing_v2 is False

    def test_empty_secret_string_becomes_none(self) -> None:
        s = _settings(groq_api_key="")
        assert s.groq_api_key is None


class TestSettingsOverride:
    """Settings should accept valid overrides passed as keyword arguments."""

    def test_override_environment(self) -> None:
        s = _settings(environment="production")
        assert s.environment == Environment.production

    def test_override_llm_provider(self) -> None:
        s = _settings(llm_provider="ollama")
        assert s.llm_provider == LLMProvider.ollama

    def test_override_llm_model(self) -> None:
        s = _settings(llm_model="llama3.1")
        assert s.llm_model == "llama3.1"

    def test_set_groq_api_key(self) -> None:
        s = _settings(groq_api_key="sk-test-key")
        assert s.groq_api_key is not None
        # SecretStr should NOT expose the value via str()
        assert "sk-test-key" not in str(s.groq_api_key)
        # The actual value must be retrieved explicitly
        assert s.groq_api_key.get_secret_value() == "sk-test-key"


class TestSettingsValidation:
    """Settings should reject invalid enum values with a clear error."""

    def test_invalid_environment_raises(self) -> None:
        with pytest.raises(ValidationError):
            _settings(environment="invalid_env")

    def test_invalid_llm_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            _settings(llm_provider="anthropic")


class TestSecretFieldSafety:
    """SecretStr fields must not leak their values through repr or str."""

    def test_groq_api_key_repr_hides_value(self) -> None:
        s = _settings(groq_api_key="super-secret")
        assert "super-secret" not in repr(s)
