"""
Unit tests for prompt template utilities.

Tests:
1. build_chat_prompt — template construction and metadata
2. format_to_chat_messages — variable interpolation and role mapping
3. _lc_type_to_role — mapping table completeness and error handling
"""

import pytest
from langchain_core.prompts import ChatPromptTemplate

from rag_assistant.llm.interface import MessageRole
from rag_assistant.llm.prompt import _lc_type_to_role, build_chat_prompt, format_to_chat_messages

# ---------------------------------------------------------------------------
# 1. build_chat_prompt
# ---------------------------------------------------------------------------


class TestBuildChatPrompt:
    def test_returns_chat_prompt_template(self) -> None:
        prompt = build_chat_prompt("You are helpful.", "Answer {question}.")
        assert isinstance(prompt, ChatPromptTemplate)

    def test_has_two_messages(self) -> None:
        prompt = build_chat_prompt("System.", "User {x}.")
        assert len(prompt.messages) == 2

    def test_input_variables_detected(self) -> None:
        prompt = build_chat_prompt("System.", "Explain {concept} for {level}.")
        variables = set(prompt.input_variables)
        assert variables == {"concept", "level"}

    def test_no_variables_in_system_message(self) -> None:
        prompt = build_chat_prompt("Fixed system.", "Hello {name}.")
        # System message has no variables — only user template does.
        assert "name" in prompt.input_variables

    def test_no_template_variables(self) -> None:
        prompt = build_chat_prompt("System.", "Fixed user message.")
        assert prompt.input_variables == []

    def test_format_messages_returns_two_messages(self) -> None:
        prompt = build_chat_prompt("System.", "Hello {name}.")
        messages = prompt.format_messages(name="World")
        assert len(messages) == 2


# ---------------------------------------------------------------------------
# 2. format_to_chat_messages
# ---------------------------------------------------------------------------


class TestFormatToChatMessages:
    def test_returns_list_of_chat_messages(self) -> None:
        from rag_assistant.llm.interface import ChatMessage

        prompt = build_chat_prompt("System.", "Hello {name}.")
        result = format_to_chat_messages(prompt, name="World")
        assert all(isinstance(m, ChatMessage) for m in result)

    def test_system_role_mapped_correctly(self) -> None:
        prompt = build_chat_prompt("System content.", "User {x}.")
        result = format_to_chat_messages(prompt, x="value")
        assert result[0].role == MessageRole.system

    def test_user_role_mapped_correctly(self) -> None:
        prompt = build_chat_prompt("System.", "User {x}.")
        result = format_to_chat_messages(prompt, x="value")
        assert result[1].role == MessageRole.user

    def test_system_content_preserved(self) -> None:
        prompt = build_chat_prompt("I am a tutor.", "Question: {q}.")
        result = format_to_chat_messages(prompt, q="What is RAG?")
        assert result[0].content == "I am a tutor."

    def test_user_content_interpolated(self) -> None:
        prompt = build_chat_prompt("System.", "Explain {topic}.")
        result = format_to_chat_messages(prompt, topic="embeddings")
        assert result[1].content == "Explain embeddings."

    def test_multiple_variables_interpolated(self) -> None:
        prompt = build_chat_prompt("System.", "Compare {a} with {b}.")
        result = format_to_chat_messages(prompt, a="BM25", b="dense retrieval")
        assert "BM25" in result[1].content
        assert "dense retrieval" in result[1].content

    def test_returns_two_messages(self) -> None:
        prompt = build_chat_prompt("System.", "User {x}.")
        result = format_to_chat_messages(prompt, x="v")
        assert len(result) == 2

    def test_missing_variable_raises(self) -> None:
        prompt = build_chat_prompt("System.", "Hello {name}.")
        with pytest.raises(KeyError):
            format_to_chat_messages(prompt)  # 'name' not provided

    def test_result_usable_as_chat_input(self) -> None:
        """Messages produced by the bridge satisfy the ChatMessage type."""
        from rag_assistant.llm.interface import ChatMessage

        prompt = build_chat_prompt("You are a tutor.", "What is {concept}?")
        messages = format_to_chat_messages(prompt, concept="RAG")
        assert all(isinstance(m, ChatMessage) for m in messages)
        assert all(m.role in MessageRole for m in messages)


# ---------------------------------------------------------------------------
# 3. _lc_type_to_role
# ---------------------------------------------------------------------------


class TestLcTypeToRole:
    def test_system_maps_to_system(self) -> None:
        assert _lc_type_to_role("system") == MessageRole.system

    def test_human_maps_to_user(self) -> None:
        assert _lc_type_to_role("human") == MessageRole.user

    def test_ai_maps_to_assistant(self) -> None:
        assert _lc_type_to_role("ai") == MessageRole.assistant

    def test_unknown_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Cannot convert"):
            _lc_type_to_role("function")

    def test_tool_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Cannot convert"):
            _lc_type_to_role("tool")

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _lc_type_to_role("")
