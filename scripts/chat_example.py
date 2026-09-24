"""
Minimal chat example.

Demonstrates the full path:
  Settings (from .env) → create_llm_provider → ChatMessage → LLMResponse

Usage
-----
    uv run python scripts/chat_example.py

Requirements
------------
Set GROQ_API_KEY in your .env file (or as an environment variable).
LLM_PROVIDER defaults to "groq".
LLM_MODEL defaults to "qwen/qwen3.8-27b".
"""

import logging
import sys

from rag_assistant.config import settings
from rag_assistant.llm import ChatMessage, MessageRole, create_llm_provider

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(levelname)s %(name)s — %(message)s",
)


def main() -> None:
    print(f"Provider : {settings.llm_provider}")
    print(f"Model    : {settings.llm_model}")
    print(f"Env      : {settings.environment}")
    print()

    # Build the provider from configuration — no provider name in application code.
    try:
        provider = create_llm_provider(settings)
    except ValueError as exc:
        print(f"[error] Could not initialise provider: {exc}", file=sys.stderr)
        print("Make sure GROQ_API_KEY is set in your .env file.", file=sys.stderr)
        sys.exit(1)

    messages = [
        ChatMessage(
            role=MessageRole.system,
            content=("You are a concise AI Engineering tutor. Answer in three sentences or fewer."),
        ),
        ChatMessage(
            role=MessageRole.user,
            content="What is retrieval-augmented generation and why does it reduce hallucination?",
        ),
    ]

    print("Sending request...")
    result = provider.chat(messages)

    print()
    print("-" * 60)
    print(result.content)
    print("-" * 60)
    print(f"Model    : {result.model}")
    print(f"Latency  : {result.latency_ms:.0f} ms")
    print(
        f"Tokens   : {result.usage.prompt_tokens} prompt + "
        f"{result.usage.completion_tokens} completion = "
        f"{result.usage.total_tokens} total"
    )


if __name__ == "__main__":
    main()
