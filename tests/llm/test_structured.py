"""
Tests for the structured LLM output layer.

Test coverage
-------------
extract_json
  - raw text is already valid JSON          → returned unchanged
  - JSON wrapped in ```json...```           → extracted
  - JSON wrapped in ``` ... ```             → extracted
  - JSON embedded after prose               → extracted
  - no JSON-like content                    → JSONParseError

parse_structured
  - valid JSON matching schema              → returns model instance
  - JSON with extra fields                  → extra fields ignored
  - JSON missing a required field           → SchemaValidationError
  - JSON with wrong enum value              → SchemaValidationError
  - JSON with wrong type for a field        → SchemaValidationError
  - JSON-like but malformed syntax          → JSONParseError
  - plain prose (no JSON)                   → JSONParseError

build_json_system_prompt
  - returns a non-empty string              → sanity
  - contains schema field names             → schema is embedded

_build_messages
  - no system message present              → system message injected at front
  - system message present                 → JSON instruction appended to it

call_structured (mocked provider)
  - provider returns valid JSON             → returns model instance
  - provider returns invalid JSON           → raises JSONParseError
  - provider returns JSON failing schema    → raises SchemaValidationError

acall_structured (mocked provider)
  - same contract as call_structured        → async path verified

What is NOT tested here
-----------------------
- Real LLM calls.  Those belong in integration tests.
- Retry or fallback logic (service-layer concern, not implemented here).
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_assistant.llm.interface import ChatMessage, LLMResponse, MessageRole, TokenUsage
from rag_assistant.llm.structured import (
    JSONParseError,
    SchemaValidationError,
    StructuredOutputError,
    _build_messages,
    acall_structured,
    build_json_system_prompt,
    call_structured,
    extract_json,
    parse_structured,
)
from rag_assistant.schemas.question_analysis import (
    Difficulty,
    QuestionAnalysis,
    QuestionIntent,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_PAYLOAD: dict[str, object] = {
    "intent": "conceptual",
    "difficulty": "intermediate",
    "concepts": ["RAG", "vector similarity", "embeddings"],
    "answer": "RAG grounds generation in retrieved documents, reducing hallucination.",
}


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test-model",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        latency_ms=50.0,
    )


def _make_provider(content: str) -> MagicMock:
    """Return a mock provider whose chat() returns the given content."""
    mock = MagicMock()
    mock.chat.return_value = _make_llm_response(content)
    return mock


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_already_valid_json_object(self) -> None:
        raw = json.dumps(VALID_PAYLOAD)
        assert extract_json(raw) == raw

    def test_json_in_fenced_block_with_language_tag(self) -> None:
        raw = f"```json\n{json.dumps(VALID_PAYLOAD)}\n```"
        extracted = extract_json(raw)
        # Must be parseable
        assert json.loads(extracted) == VALID_PAYLOAD

    def test_json_in_fenced_block_without_language_tag(self) -> None:
        raw = f"```\n{json.dumps(VALID_PAYLOAD)}\n```"
        extracted = extract_json(raw)
        assert json.loads(extracted) == VALID_PAYLOAD

    def test_json_after_prose(self) -> None:
        raw = f"Here is the analysis you requested:\n{json.dumps(VALID_PAYLOAD)}"
        extracted = extract_json(raw)
        assert json.loads(extracted) == VALID_PAYLOAD

    def test_json_surrounded_by_prose(self) -> None:
        raw = f"Result: {json.dumps(VALID_PAYLOAD)} Hope this helps!"
        extracted = extract_json(raw)
        assert json.loads(extracted) == VALID_PAYLOAD

    def test_no_json_raises(self) -> None:
        with pytest.raises(JSONParseError, match="No JSON object found"):
            extract_json("This is just plain text with no JSON anywhere.")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(JSONParseError):
            extract_json("")

    def test_json_parse_error_stores_raw(self) -> None:
        raw = "totally plain text"
        with pytest.raises(JSONParseError) as exc_info:
            extract_json(raw)
        assert exc_info.value.raw == raw


# ---------------------------------------------------------------------------
# parse_structured
# ---------------------------------------------------------------------------


class TestParseStructured:
    def test_valid_json_returns_model(self) -> None:
        result = parse_structured(json.dumps(VALID_PAYLOAD), QuestionAnalysis)
        assert isinstance(result, QuestionAnalysis)
        assert result.intent == QuestionIntent.conceptual
        assert result.difficulty == Difficulty.intermediate
        assert result.concepts == ["RAG", "vector similarity", "embeddings"]
        assert "hallucination" in result.answer

    def test_extra_fields_are_ignored(self) -> None:
        payload = {**VALID_PAYLOAD, "extra_key": "some_value", "another_key": 42}
        result = parse_structured(json.dumps(payload), QuestionAnalysis)
        assert isinstance(result, QuestionAnalysis)
        # extra_key must not appear on the model
        assert not hasattr(result, "extra_key")

    def test_missing_required_field_raises(self) -> None:
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "answer"}
        with pytest.raises(SchemaValidationError) as exc_info:
            parse_structured(json.dumps(payload), QuestionAnalysis)
        assert exc_info.value.cause is not None  # wraps ValidationError

    def test_wrong_enum_value_raises(self) -> None:
        payload = {**VALID_PAYLOAD, "difficulty": "expert"}  # not in Difficulty enum
        with pytest.raises(SchemaValidationError, match="QuestionAnalysis"):
            parse_structured(json.dumps(payload), QuestionAnalysis)

    def test_wrong_type_for_concepts_raises(self) -> None:
        # concepts must be a list, not a string
        payload = {**VALID_PAYLOAD, "concepts": "RAG, embeddings"}
        with pytest.raises(SchemaValidationError):
            parse_structured(json.dumps(payload), QuestionAnalysis)

    def test_malformed_json_raises_json_parse_error(self) -> None:
        with pytest.raises(JSONParseError):
            parse_structured('{"intent": "conceptual", "difficulty":}', QuestionAnalysis)

    def test_plain_prose_raises_json_parse_error(self) -> None:
        with pytest.raises(JSONParseError):
            parse_structured("The answer is retrieval-augmented generation.", QuestionAnalysis)

    def test_schema_validation_error_stores_raw(self) -> None:
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "answer"}
        raw = json.dumps(payload)
        with pytest.raises(SchemaValidationError) as exc_info:
            parse_structured(raw, QuestionAnalysis)
        assert exc_info.value.raw == raw

    def test_schema_validation_error_is_structured_output_error(self) -> None:
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "answer"}
        with pytest.raises(StructuredOutputError):
            parse_structured(json.dumps(payload), QuestionAnalysis)

    def test_json_parse_error_is_structured_output_error(self) -> None:
        with pytest.raises(StructuredOutputError):
            parse_structured("not json", QuestionAnalysis)

    def test_markdown_fenced_json_is_parsed(self) -> None:
        raw = f"```json\n{json.dumps(VALID_PAYLOAD)}\n```"
        result = parse_structured(raw, QuestionAnalysis)
        assert isinstance(result, QuestionAnalysis)


# ---------------------------------------------------------------------------
# build_json_system_prompt
# ---------------------------------------------------------------------------


class TestBuildJsonSystemPrompt:
    def test_returns_non_empty_string(self) -> None:
        prompt = build_json_system_prompt(QuestionAnalysis)
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_contains_field_names(self) -> None:
        prompt = build_json_system_prompt(QuestionAnalysis)
        assert "intent" in prompt
        assert "difficulty" in prompt
        assert "concepts" in prompt
        assert "answer" in prompt

    def test_contains_enum_values(self) -> None:
        prompt = build_json_system_prompt(QuestionAnalysis)
        assert "conceptual" in prompt
        assert "beginner" in prompt

    def test_instructs_json_only(self) -> None:
        prompt = build_json_system_prompt(QuestionAnalysis)
        assert "JSON" in prompt


# ---------------------------------------------------------------------------
# _build_messages (internal helper)
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_injects_system_message_when_none_present(self) -> None:
        messages = [ChatMessage(role=MessageRole.user, content="What is RAG?")]
        result = _build_messages(messages, QuestionAnalysis)
        assert result[0].role == MessageRole.system
        assert result[-1].role == MessageRole.user
        assert result[-1].content == "What is RAG?"

    def test_system_message_contains_schema(self) -> None:
        messages = [ChatMessage(role=MessageRole.user, content="?")]
        result = _build_messages(messages, QuestionAnalysis)
        assert "intent" in result[0].content

    def test_appends_to_existing_system_message(self) -> None:
        messages = [
            ChatMessage(role=MessageRole.system, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.user, content="What is RAG?"),
        ]
        result = _build_messages(messages, QuestionAnalysis)
        # Still one system message
        system_msgs = [m for m in result if m.role == MessageRole.system]
        assert len(system_msgs) == 1
        # Original content is preserved
        assert "You are a helpful assistant." in system_msgs[0].content
        # JSON instruction was appended
        assert "JSON" in system_msgs[0].content

    def test_message_count_preserved_without_system(self) -> None:
        messages = [ChatMessage(role=MessageRole.user, content="?")]
        result = _build_messages(messages, QuestionAnalysis)
        # 1 user + 1 injected system
        assert len(result) == 2

    def test_message_count_preserved_with_system(self) -> None:
        messages = [
            ChatMessage(role=MessageRole.system, content="Context."),
            ChatMessage(role=MessageRole.user, content="?"),
        ]
        result = _build_messages(messages, QuestionAnalysis)
        # merged into 1 system + 1 user = 2
        assert len(result) == 2


# ---------------------------------------------------------------------------
# call_structured
# ---------------------------------------------------------------------------


class TestCallStructured:
    def test_valid_response_returns_model(self) -> None:
        provider = _make_provider(json.dumps(VALID_PAYLOAD))
        messages = [ChatMessage(role=MessageRole.user, content="What is RAG?")]

        result = call_structured(provider, messages, QuestionAnalysis)

        assert isinstance(result, QuestionAnalysis)
        assert result.intent == QuestionIntent.conceptual

    def test_provider_chat_is_called_once(self) -> None:
        provider = _make_provider(json.dumps(VALID_PAYLOAD))
        messages = [ChatMessage(role=MessageRole.user, content="?")]

        call_structured(provider, messages, QuestionAnalysis)

        provider.chat.assert_called_once()

    def test_system_prompt_is_injected(self) -> None:
        provider = _make_provider(json.dumps(VALID_PAYLOAD))
        messages = [ChatMessage(role=MessageRole.user, content="?")]

        call_structured(provider, messages, QuestionAnalysis)

        sent_messages: list[ChatMessage] = provider.chat.call_args[0][0]
        system_msgs = [m for m in sent_messages if m.role == MessageRole.system]
        assert len(system_msgs) == 1
        assert "intent" in system_msgs[0].content

    def test_invalid_json_raises(self) -> None:
        provider = _make_provider("Sorry, I cannot help with that.")
        messages = [ChatMessage(role=MessageRole.user, content="?")]

        with pytest.raises(JSONParseError):
            call_structured(provider, messages, QuestionAnalysis)

    def test_schema_violation_raises(self) -> None:
        bad_payload = {**VALID_PAYLOAD, "difficulty": "legendary"}
        provider = _make_provider(json.dumps(bad_payload))
        messages = [ChatMessage(role=MessageRole.user, content="?")]

        with pytest.raises(SchemaValidationError):
            call_structured(provider, messages, QuestionAnalysis)

    def test_markdown_wrapped_json_is_handled(self) -> None:
        raw = f"```json\n{json.dumps(VALID_PAYLOAD)}\n```"
        provider = _make_provider(raw)
        messages = [ChatMessage(role=MessageRole.user, content="?")]

        result = call_structured(provider, messages, QuestionAnalysis)
        assert isinstance(result, QuestionAnalysis)


# ---------------------------------------------------------------------------
# acall_structured
# ---------------------------------------------------------------------------


class TestACallStructured:
    def test_async_valid_response_returns_model(self) -> None:
        provider = MagicMock()
        provider.achat = AsyncMock(return_value=_make_llm_response(json.dumps(VALID_PAYLOAD)))
        messages = [ChatMessage(role=MessageRole.user, content="What is RAG?")]

        result = asyncio.run(acall_structured(provider, messages, QuestionAnalysis))

        assert isinstance(result, QuestionAnalysis)
        assert result.difficulty == Difficulty.intermediate

    def test_async_invalid_json_raises(self) -> None:
        provider = MagicMock()
        provider.achat = AsyncMock(return_value=_make_llm_response("plain text"))
        messages = [ChatMessage(role=MessageRole.user, content="?")]

        with pytest.raises(JSONParseError):
            asyncio.run(acall_structured(provider, messages, QuestionAnalysis))

    def test_async_uses_achat(self) -> None:
        provider = MagicMock()
        provider.achat = AsyncMock(return_value=_make_llm_response(json.dumps(VALID_PAYLOAD)))
        messages = [ChatMessage(role=MessageRole.user, content="?")]

        asyncio.run(acall_structured(provider, messages, QuestionAnalysis))

        provider.achat.assert_called_once()
        provider.chat.assert_not_called()
