"""
Live integration test — Groq + Ollama.

Tests the full stack:
  1. Provider init from settings
  2. Standard chat (provider.chat)
  3. Structured output (call_structured)
  4. Tool calling (run_with_tools) with the calculator

Usage
-----
    uv run python scripts/live_test.py               # Groq only
    uv run python scripts/live_test.py --all         # Groq + Ollama

Prerequisites
-------------
- GROQ_API_KEY set in .env
- Ollama running at OLLAMA_BASE_URL with a model pulled
  (e.g. `ollama pull qwen2.5:0.5b`)
"""

import sys
import time
import traceback

from rag_assistant.config import Settings
from rag_assistant.llm import (
    ChatMessage,
    LLMProviderProtocol,
    MessageRole,
    call_structured,
    create_llm_provider,
)
from rag_assistant.schemas.question_analysis import QuestionAnalysis
from rag_assistant.tools import CalculatorTool, DocSearchTool, ToolRegistry, run_with_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OK = "[OK]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

def section(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check(label: str, passed: bool, detail: str = "") -> None:
    mark = OK if passed else FAIL
    line = f"  {mark}  {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------


def test_chat(provider: LLMProviderProtocol, provider_name: str) -> None:
    """Standard chat: 3-sentence answer about RAG."""
    print(f"\n--- chat ({provider_name}) ---")
    messages = [
        ChatMessage(
            role=MessageRole.system,
            content="You are a concise AI tutor. Answer in one sentence only.",
        ),
        ChatMessage(
            role=MessageRole.user,
            content="What is the main benefit of retrieval-augmented generation?",
        ),
    ]
    t0 = time.perf_counter()
    try:
        result = provider.chat(messages)
        elapsed = (time.perf_counter() - t0) * 1000
        check("chat() returned LLMResponse", True)
        check("content is non-empty", bool(result.content))
        check("latency tracked", result.latency_ms > 0, f"{result.latency_ms:.0f} ms")
        check("tokens tracked", result.usage.total_tokens > 0, f"{result.usage.total_tokens} tokens")
        print(f'\n  Response: "{result.content[:120]}"')
    except Exception as exc:
        check("chat() succeeded", False, str(exc)[:100])


def test_structured(provider: LLMProviderProtocol, provider_name: str) -> None:
    """Structured output: extract QuestionAnalysis from a user question."""
    print(f"\n--- structured output ({provider_name}) ---")
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content="What is the difference between dense retrieval and BM25 for RAG?",
        ),
    ]
    try:
        result = call_structured(provider, messages, QuestionAnalysis)
        check("call_structured() returned QuestionAnalysis", True)
        check("intent is valid enum", bool(result.intent))
        check("difficulty is valid enum", bool(result.difficulty))
        check("concepts is non-empty list", bool(result.concepts))
        check("answer is non-empty", bool(result.answer))
        print(f"  Intent    : {result.intent}")
        print(f"  Difficulty: {result.difficulty}")
        print(f"  Concepts  : {result.concepts[:3]}")
        print(f"  Answer    : {result.answer[:120]}")
    except Exception as exc:
        check("call_structured() succeeded", False, str(exc)[:120])
        traceback.print_exc()


def test_tool_calling(provider: LLMProviderProtocol, provider_name: str) -> None:
    """Tool calling: ask the model to compute an expression."""
    print(f"\n--- tool calling ({provider_name}) ---")
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(DocSearchTool())

    messages = [
        ChatMessage(
            role=MessageRole.system,
            content=(
                "You are a helpful assistant with access to tools. "
                "Use the calculator tool to answer math questions precisely."
            ),
        ),
        ChatMessage(
            role=MessageRole.user,
            content="What is 847 multiplied by 23? Use the calculator.",
        ),
    ]
    try:
        result = run_with_tools(provider, registry, messages)
        check("run_with_tools() returned LLMResponse", True)
        check("content is non-empty", bool(result.content))
        answer_has_number = "19481" in result.content.replace(",", "").replace(".", "")
        check(
            "answer contains correct result (19481)",
            answer_has_number,
            f'Got: "{result.content[:120]}"',
        )
        print(f'\n  Response: "{result.content[:160]}"')
    except Exception as exc:
        check("run_with_tools() succeeded", False, str(exc)[:120])
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_for_provider(label: str, model: str, provider_str: str, extra: dict | None = None) -> None:
    section(f"Provider: {label}  (model: {model})")
    overrides = {"llm_provider": provider_str, "llm_model": model}
    if extra:
        overrides.update(extra)
    try:
        s = Settings(_env_file=".env", **overrides)
        provider = create_llm_provider(s)
        print(f"  Initialised: {provider!r}")
    except Exception as exc:
        print(f"  {FAIL}  Could not initialise provider: {exc}")
        return

    test_chat(provider, label)
    test_structured(provider, label)
    test_tool_calling(provider, label)


def main() -> None:
    run_all = "--all" in sys.argv

    # --- Groq ---
    run_for_provider(
        label="Groq",
        model="qwen/qwen3.8-27b",
        provider_str="groq",
    )

    # --- Ollama ---
    if run_all:
        run_for_provider(
            label="Ollama",
            model="qwen2.5:0.5b",
            provider_str="ollama",
            extra={"ollama_base_url": "http://localhost:11434"},
        )
    else:
        section("Provider: Ollama  (skipped — run with --all to include)")
        print(f"  {SKIP}  Pass --all to run Ollama tests.")

    print()
    print("=" * 60)
    print("  Done.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
