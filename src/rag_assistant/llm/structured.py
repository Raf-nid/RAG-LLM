"""
Structured LLM output utilities.

This module bridges the probabilistic world of LLM text generation and the
deterministic world of typed, validated application data.

Architecture
------------
The flow for one structured call:

    messages + schema
         ↓
    build_json_system_prompt(schema)   → inject as system message
         ↓
    provider.chat(messages)            → LLMResponse (raw string content)
         ↓
    extract_json(content)              → isolated JSON text
         ↓
    json.loads(...)                    → Python dict
         ↓
    Model.model_validate(...)          → typed Pydantic instance
         ↓
    caller receives T

Every step that can fail raises a specific subclass of ``StructuredOutputError``
so callers can handle different failure modes differently.

Key design property
-------------------
The Pydantic schemas used here are plain domain models.  They have no
dependency on any provider, LangChain, or this module.  The calling code
can define schemas in ``rag_assistant.schemas.*`` and remain entirely
unaware of how they are requested from the LLM.

What this module does NOT do
-----------------------------
- It does not retry failed requests.
- It does not fall back to a different model.
- It does not stream tokens.
- It does not use provider-native structured-output APIs (e.g. Groq's
  ``response_format`` or LangChain's ``with_structured_output``).

All of these are valid extensions but belong at the service layer, not here.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, ValidationError

from .interface import ChatMessage, LLMProviderProtocol, MessageRole

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StructuredOutputError(Exception):
    """
    Base class for all structured-output parsing failures.

    Catching this class handles every failure mode.  Catch the subclasses
    when you need to distinguish JSON syntax errors from schema violations.
    """


class JSONParseError(StructuredOutputError):
    """
    The LLM response could not be parsed as JSON.

    This usually means:
    - The model ignored the JSON instruction entirely.
    - The model returned valid prose with a badly-formed JSON fragment.
    - The model produced truncated JSON (hit a token limit mid-object).

    Retrying or using a larger model may help.
    """

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class SchemaValidationError(StructuredOutputError):
    """
    The LLM returned parseable JSON that does not match the expected schema.

    This usually means:
    - A required field is missing.
    - A field has the wrong type.
    - An enum field contains a value not in the allowed set.

    Retrying with a more explicit prompt or a stricter JSON schema in the
    system prompt may help.  Switching to a larger model also tends to
    improve compliance.
    """

    def __init__(self, message: str, raw: str, cause: ValidationError) -> None:
        super().__init__(message)
        self.raw = raw
        self.cause = cause


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

# Match a fenced code block: ```json ... ``` or ``` ... ```
_FENCED_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json(text: str) -> str:
    """
    Attempt to isolate a JSON object (or array) from raw LLM output.

    LLMs frequently wrap JSON in markdown code fences or precede it with
    explanatory prose.  This function tries several strategies in order:

    1. The raw text is already valid JSON — return as-is.
    2. The text contains a fenced code block (```json...```) — extract content.
    3. The text contains a bare JSON object (first ``{`` to matching ``}``) or
       array (first ``[`` to matching ``]``) — extract that substring.

    Parameters
    ----------
    text:
        Raw string from the LLM.

    Returns
    -------
    str
        A string that may be valid JSON.  Callers must still call
        ``json.loads`` to confirm.

    Raises
    ------
    JSONParseError
        If no JSON-like content can be found.
    """
    stripped = text.strip()

    # Strategy 1: already valid JSON
    if stripped.startswith(("{", "[")):
        return stripped

    # Strategy 2: fenced code block
    match = _FENCED_BLOCK.search(stripped)
    if match:
        return match.group(1).strip()

    # Strategy 3: find first { ... } or [ ... ]
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = stripped.find(open_char)
        end = stripped.rfind(close_char)
        if start != -1 and end > start:
            return stripped[start : end + 1]

    raise JSONParseError(
        f"No JSON object found in LLM response (length={len(text)}). "
        "The model may have ignored the structured-output instruction.",
        raw=text,
    )


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def parse_structured[T: BaseModel](raw: str, model: type[T]) -> T:
    """
    Parse raw LLM output into a validated Pydantic model instance.

    Parameters
    ----------
    raw:
        Raw string returned by the LLM provider.
    model:
        Pydantic ``BaseModel`` subclass to validate against.

    Returns
    -------
    T
        A validated instance of ``model``.

    Raises
    ------
    JSONParseError
        If the text cannot be parsed as JSON.
    SchemaValidationError
        If the JSON does not match ``model``'s schema.
    """
    # Step 1 — extract candidate JSON string
    candidate = extract_json(raw)

    # Step 2 — parse JSON
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JSONParseError(
            f"JSON syntax error in LLM response: {exc}",
            raw=raw,
        ) from exc

    # Step 3 — validate with Pydantic
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise SchemaValidationError(
            f"LLM response does not match {model.__name__}: "
            f"{exc.error_count()} validation error(s)",
            raw=raw,
            cause=exc,
        ) from exc


# ---------------------------------------------------------------------------
# System prompt generation
# ---------------------------------------------------------------------------


def build_json_system_prompt(schema: type[BaseModel]) -> str:
    """
    Build a system prompt that instructs the LLM to respond with JSON
    matching ``schema``.

    The schema's JSON Schema is embedded verbatim so the model has the full
    type information including enums, required fields, and array types.

    Parameters
    ----------
    schema:
        A Pydantic ``BaseModel`` subclass.

    Returns
    -------
    str
        A system prompt string ready to be sent as a ``MessageRole.system``
        message.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    return (
        "You are a precise assistant that responds ONLY with valid JSON.\n\n"
        "Your response must be a single JSON object matching this schema exactly:\n\n"
        f"{schema_json}\n\n"
        "Rules:\n"
        "- Respond with ONLY the JSON object. No prose, no explanation.\n"
        "- Do NOT wrap the JSON in markdown code fences.\n"
        "- All required fields must be present.\n"
        "- Use only the allowed enum values where specified.\n"
    )


# ---------------------------------------------------------------------------
# Structured callers
# ---------------------------------------------------------------------------


def _build_messages(
    messages: list[ChatMessage],
    schema: type[BaseModel],
) -> list[ChatMessage]:
    """
    Inject a JSON-format system instruction into the message list.

    If a system message already exists, the JSON instruction is appended to
    its content so both instructions are visible to the model.  If no system
    message exists, one is prepended.
    """
    json_instruction = build_json_system_prompt(schema)

    system_messages = [m for m in messages if m.role == MessageRole.system]
    other_messages = [m for m in messages if m.role != MessageRole.system]

    if system_messages:
        combined = system_messages[0].content + "\n\n" + json_instruction
        merged_system = ChatMessage(role=MessageRole.system, content=combined)
        return [merged_system, *other_messages]

    injected = ChatMessage(role=MessageRole.system, content=json_instruction)
    return [injected, *messages]


def call_structured[T: BaseModel](
    provider: LLMProviderProtocol,
    messages: list[ChatMessage],
    schema: type[T],
) -> T:
    """
    Call the provider synchronously and return a validated structured response.

    The JSON-format system prompt is injected automatically.  Callers do not
    need to add any JSON instructions to their messages.

    Parameters
    ----------
    provider:
        Any object satisfying ``LLMProviderProtocol``.
    messages:
        Conversation messages.  Typically a single user message is sufficient.
        A system message may be included; the JSON instruction will be appended
        to it rather than replacing it.
    schema:
        Pydantic model class to validate the response against.

    Returns
    -------
    T
        A validated instance of ``schema``.

    Raises
    ------
    JSONParseError
        If the LLM response cannot be parsed as JSON.
    SchemaValidationError
        If the JSON does not match ``schema``.
    """
    full_messages = _build_messages(messages, schema)
    logger.debug("call_structured — schema=%s, messages=%d", schema.__name__, len(full_messages))

    response = provider.chat(full_messages)

    logger.debug(
        "call_structured — received %d chars (latency=%.0fms)",
        len(response.content),
        response.latency_ms,
    )

    return parse_structured(response.content, schema)


async def acall_structured[T: BaseModel](
    provider: LLMProviderProtocol,
    messages: list[ChatMessage],
    schema: type[T],
) -> T:
    """
    Async variant of ``call_structured``.  Identical contract, async execution.
    """
    full_messages = _build_messages(messages, schema)
    logger.debug("acall_structured — schema=%s, messages=%d", schema.__name__, len(full_messages))

    response = await provider.achat(full_messages)

    logger.debug(
        "acall_structured — received %d chars (latency=%.0fms)",
        len(response.content),
        response.latency_ms,
    )

    return parse_structured(response.content, schema)
