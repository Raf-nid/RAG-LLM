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


class TestSettingsDefaults:
    """Settings should load safely with no environment variables set."""

    def test_default_environment(self) -> None:
        s = Settings()
        assert s.environment == Environment.development

    def test_default_llm_provider(self) -> None:
        s = Settings()
        assert s.llm_provider == LLMProvider.groq

    def test_default_llm_model(self) -> None:
        s = Settings()
        assert s.llm_model == "llama-3.3-70b-versatile"

    def test_no_api_keys_by_default(self) -> None:
        s = Settings()
        assert s.groq_api_key is None
        assert s.openai_api_key is None
        assert s.langsmith_api_key is None

    def test_langchain_tracing_disabled_by_default(self) -> None:
        s = Settings()
        assert s.langchain_tracing_v2 is False


class TestSettingsOverride:
    """Settings should accept valid overrides passed as keyword arguments."""

    def test_override_environment(self) -> None:
        s = Settings(environment="production")  # type: ignore[call-arg]
        assert s.environment == Environment.production

    def test_override_llm_provider(self) -> None:
        s = Settings(llm_provider="openai")  # type: ignore[call-arg]
        assert s.llm_provider == LLMProvider.openai

    def test_override_llm_model(self) -> None:
        s = Settings(llm_model="gpt-4o")  # type: ignore[call-arg]
        assert s.llm_model == "gpt-4o"

    def test_set_groq_api_key(self) -> None:
        s = Settings(groq_api_key="sk-test-key")  # type: ignore[call-arg]
        assert s.groq_api_key is not None
        # SecretStr should NOT expose the value via str()
        assert "sk-test-key" not in str(s.groq_api_key)
        # The actual value must be retrieved explicitly
        assert s.groq_api_key.get_secret_value() == "sk-test-key"


class TestSettingsValidation:
    """Settings should reject invalid enum values with a clear error."""

    def test_invalid_environment_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(environment="invalid_env")  # type: ignore[call-arg]

    def test_invalid_llm_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(llm_provider="anthropic")  # type: ignore[call-arg]


class TestSecretFieldSafety:
    """SecretStr fields must not leak their values through repr or str."""

    def test_groq_api_key_repr_hides_value(self) -> None:
        s = Settings(groq_api_key="super-secret")  # type: ignore[call-arg]
        assert "super-secret" not in repr(s)

    def test_openai_api_key_repr_hides_value(self) -> None:
        s = Settings(openai_api_key="super-secret")  # type: ignore[call-arg]
        assert "super-secret" not in repr(s)
