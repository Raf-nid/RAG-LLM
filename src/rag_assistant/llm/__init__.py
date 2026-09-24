"""
LLM provider abstraction.

Public API of this module — import from here, not from submodules:

    from rag_assistant.llm import create_llm_provider, ChatMessage, LLMResponse

The concrete provider classes (GroqProvider, etc.) are intentionally
not re-exported.  They are implementation details.
"""

from .factory import create_llm_provider
from .interface import ChatMessage, LLMProviderProtocol, LLMResponse, MessageRole, TokenUsage

__all__ = [
    "ChatMessage",
    "LLMProviderProtocol",
    "LLMResponse",
    "MessageRole",
    "TokenUsage",
    "create_llm_provider",
]
