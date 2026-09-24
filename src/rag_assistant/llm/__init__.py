"""
LLM package public API.

Import these names; never import directly from sub-modules.

Layers
------
interface     — pure Python domain types and Protocol (no LangChain)
groq / ollama — LangChain-backed providers (ChatGroq / ChatOllama)
factory       — provider construction from Settings
structured    — prompt-injection structured output (works with any provider)
prompt        — ChatPromptTemplate builders and bridge to domain types
chain         — LCEL chain builders and native structured output
"""

from .chain import (
    acall_structured_native,
    build_chat_chain,
    build_structured_chain,
    call_structured_native,
)
from .factory import create_llm_provider
from .interface import (
    ChatMessage,
    LLMProviderProtocol,
    LLMResponse,
    MessageRole,
    TokenUsage,
    ToolCallingResponse,
    ToolCallRequest,
    ToolSchema,
)
from .prompt import build_chat_prompt, format_to_chat_messages
from .structured import (
    JSONParseError,
    SchemaValidationError,
    StructuredOutputError,
    acall_structured,
    call_structured,
    parse_structured,
)

__all__ = [
    "ChatMessage",
    "JSONParseError",
    "LLMProviderProtocol",
    "LLMResponse",
    "MessageRole",
    "SchemaValidationError",
    "StructuredOutputError",
    "TokenUsage",
    "ToolCallRequest",
    "ToolCallingResponse",
    "ToolSchema",
    "acall_structured",
    "acall_structured_native",
    "build_chat_chain",
    "build_chat_prompt",
    "build_structured_chain",
    "call_structured",
    "call_structured_native",
    "create_llm_provider",
    "format_to_chat_messages",
    "parse_structured",
]
