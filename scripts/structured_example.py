"""
Structured output example.

Demonstrates the full path from a user question to a validated
``QuestionAnalysis`` object:

  Settings → provider → call_structured → QuestionAnalysis

The LLM is instructed to return JSON matching the schema.
The response is validated by Pydantic before any field is accessed.

Usage
-----
    uv run python scripts/structured_example.py

Requirements
------------
Set GROQ_API_KEY in .env (or LLM_PROVIDER=ollama with Ollama running).
"""

import logging
import sys

from rag_assistant.config import settings
from rag_assistant.llm import ChatMessage, MessageRole, create_llm_provider
from rag_assistant.llm.structured import (
    JSONParseError,
    SchemaValidationError,
    call_structured,
)
from rag_assistant.schemas.question_analysis import QuestionAnalysis

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(levelname)s %(name)s - %(message)s",
)


def main() -> None:
    print(f"Provider : {settings.llm_provider}")
    print(f"Model    : {settings.llm_model}")
    print()

    try:
        provider = create_llm_provider(settings)
    except ValueError as exc:
        print(f"[error] Provider init failed: {exc}", file=sys.stderr)
        sys.exit(1)

    question = "What is the difference between dense retrieval and BM25 in RAG systems?"

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=f"Analyse this technical question: {question}",
        )
    ]

    print(f"Question : {question}")
    print("Sending structured request...")
    print()

    try:
        result: QuestionAnalysis = call_structured(provider, messages, QuestionAnalysis)
    except JSONParseError as exc:
        print("[error] LLM did not return valid JSON.", file=sys.stderr)
        print(f"        Raw response snippet: {exc.raw[:200]}", file=sys.stderr)
        sys.exit(1)
    except SchemaValidationError as exc:
        print(f"[error] LLM response did not match schema: {exc}", file=sys.stderr)
        print(f"        Details: {exc.cause}", file=sys.stderr)
        sys.exit(1)

    # All fields are guaranteed to be valid at this point.
    print("-" * 60)
    print(f"Intent     : {result.intent}")
    print(f"Difficulty : {result.difficulty}")
    print(f"Concepts   : {', '.join(result.concepts)}")
    print()
    print("Answer:")
    print(result.answer)
    print("-" * 60)


if __name__ == "__main__":
    main()
