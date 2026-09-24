"""
Tests for the QuestionAnalysis domain schema.

These tests verify the schema itself — its validation rules, defaults,
and constraint enforcement — independently of any LLM or parsing logic.

The schema is a plain Pydantic model.  These tests are deterministic:
they verify application-layer constraints, not LLM behaviour.
"""

import pytest
from pydantic import ValidationError

from rag_assistant.schemas.question_analysis import (
    Difficulty,
    QuestionAnalysis,
    QuestionIntent,
)

VALID_DATA: dict[str, object] = {
    "intent": "conceptual",
    "difficulty": "intermediate",
    "concepts": ["RAG", "embeddings"],
    "answer": "RAG retrieves relevant documents before generating an answer.",
}


class TestQuestionAnalysisValid:
    def test_constructs_from_dict(self) -> None:
        qa = QuestionAnalysis(**VALID_DATA)  # type: ignore[arg-type]
        assert qa.intent == QuestionIntent.conceptual
        assert qa.difficulty == Difficulty.intermediate
        assert qa.concepts == ["RAG", "embeddings"]

    def test_all_intent_values_accepted(self) -> None:
        for intent in QuestionIntent:
            data = {**VALID_DATA, "intent": intent.value}
            qa = QuestionAnalysis(**data)  # type: ignore[arg-type]
            assert qa.intent == intent

    def test_all_difficulty_values_accepted(self) -> None:
        for diff in Difficulty:
            data = {**VALID_DATA, "difficulty": diff.value}
            qa = QuestionAnalysis(**data)  # type: ignore[arg-type]
            assert qa.difficulty == diff

    def test_extra_fields_are_ignored(self) -> None:
        data = {**VALID_DATA, "surprise_field": "ignored", "another": 99}
        qa = QuestionAnalysis(**data)  # type: ignore[arg-type]
        assert not hasattr(qa, "surprise_field")

    def test_single_concept_is_valid(self) -> None:
        data = {**VALID_DATA, "concepts": ["embeddings"]}
        qa = QuestionAnalysis(**data)  # type: ignore[arg-type]
        assert len(qa.concepts) == 1

    def test_many_concepts_are_valid(self) -> None:
        data = {**VALID_DATA, "concepts": [f"concept_{i}" for i in range(10)]}
        qa = QuestionAnalysis(**data)  # type: ignore[arg-type]
        assert len(qa.concepts) == 10


class TestQuestionAnalysisInvalidIntent:
    def test_unknown_intent_raises(self) -> None:
        data = {**VALID_DATA, "intent": "philosophical"}
        with pytest.raises(ValidationError) as exc_info:
            QuestionAnalysis(**data)  # type: ignore[arg-type]
        assert "intent" in str(exc_info.value)

    def test_empty_intent_raises(self) -> None:
        data = {**VALID_DATA, "intent": ""}
        with pytest.raises(ValidationError):
            QuestionAnalysis(**data)  # type: ignore[arg-type]

    def test_missing_intent_raises(self) -> None:
        data = {k: v for k, v in VALID_DATA.items() if k != "intent"}
        with pytest.raises(ValidationError):
            QuestionAnalysis(**data)  # type: ignore[arg-type]


class TestQuestionAnalysisInvalidDifficulty:
    def test_unknown_difficulty_raises(self) -> None:
        data = {**VALID_DATA, "difficulty": "expert"}
        with pytest.raises(ValidationError) as exc_info:
            QuestionAnalysis(**data)  # type: ignore[arg-type]
        assert "difficulty" in str(exc_info.value)

    def test_missing_difficulty_raises(self) -> None:
        data = {k: v for k, v in VALID_DATA.items() if k != "difficulty"}
        with pytest.raises(ValidationError):
            QuestionAnalysis(**data)  # type: ignore[arg-type]


class TestQuestionAnalysisInvalidConcepts:
    def test_missing_concepts_raises(self) -> None:
        data = {k: v for k, v in VALID_DATA.items() if k != "concepts"}
        with pytest.raises(ValidationError):
            QuestionAnalysis(**data)  # type: ignore[arg-type]

    def test_empty_concepts_list_raises(self) -> None:
        data = {**VALID_DATA, "concepts": []}
        with pytest.raises(ValidationError):
            QuestionAnalysis(**data)  # type: ignore[arg-type]

    def test_concepts_as_string_raises(self) -> None:
        # LLM sometimes returns a comma-separated string instead of a list
        data = {**VALID_DATA, "concepts": "RAG, embeddings, retrieval"}
        with pytest.raises(ValidationError):
            QuestionAnalysis(**data)  # type: ignore[arg-type]


class TestQuestionAnalysisInvalidAnswer:
    def test_missing_answer_raises(self) -> None:
        data = {k: v for k, v in VALID_DATA.items() if k != "answer"}
        with pytest.raises(ValidationError):
            QuestionAnalysis(**data)  # type: ignore[arg-type]

    def test_empty_answer_raises(self) -> None:
        data = {**VALID_DATA, "answer": ""}
        with pytest.raises(ValidationError):
            QuestionAnalysis(**data)  # type: ignore[arg-type]
