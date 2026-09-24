"""
Prompt template utilities.

Why ChatPromptTemplate?
-----------------------
Manually concatenating strings into prompts has several problems:

1. Prompt structure (which messages, which roles) is scattered in application code.
2. Variable interpolation is manual and error-prone.
3. Prompts cannot be inspected, versioned, or compared independently.
4. There is no validation that required variables were provided.

``ChatPromptTemplate`` solves all of these. It defines prompts as typed
templates and separates *design* (what the prompt looks like) from *execution*
(what values fill the variables at call time).

Integration model
-----------------
This module provides two things:

1. ``build_chat_prompt`` — construct a reusable ``ChatPromptTemplate``.

2. ``format_to_chat_messages`` — fill a template and convert the result back
   to our domain ``ChatMessage`` types. This is the bridge that lets you use
   LangChain templates with our ``LLMProviderProtocol``. It keeps LangChain
   as an implementation detail rather than a protocol dependency.

If you are using LCEL chains directly (i.e. calling ``chain.invoke()``), you
do not need ``format_to_chat_messages`` — the template feeds into the chain
natively. Use the bridge only when you need ``provider.chat()`` with domain
types.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from .interface import ChatMessage, MessageRole

__all__ = [
    "build_chat_prompt",
    "format_to_chat_messages",
]


def build_chat_prompt(system: str, user_template: str) -> ChatPromptTemplate:
    """
    Build a two-message ``ChatPromptTemplate`` (system + human).

    The ``user_template`` may contain named variables in ``{variable}`` syntax.
    These are interpolated when the template is formatted.

    Parameters
    ----------
    system:
        System message content. Describes the assistant's role and behaviour.
        No variable interpolation — system messages are usually fixed.
    user_template:
        Human message template. Use ``{variable_name}`` for dynamic content.
        For example: ``"Explain {concept} for a {level} audience."``.

    Returns
    -------
    ChatPromptTemplate
        A LangChain prompt template ready for formatting or LCEL composition.

    Example
    -------
    >>> prompt = build_chat_prompt(
    ...     system="You are a concise AI tutor.",
    ...     user_template="Explain {concept} in one sentence.",
    ... )
    >>> messages = prompt.format_messages(concept="embeddings")
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", user_template),
        ]
    )


def format_to_chat_messages(
    template: ChatPromptTemplate,
    **kwargs: str,
) -> list[ChatMessage]:
    """
    Format a template with variables and return domain ``ChatMessage`` objects.

    This is the bridge between LangChain's template system and our
    ``LLMProviderProtocol`` interface.

    Use this when you want structured prompt templates but need to call
    ``provider.chat(messages)`` with our domain types rather than using
    LCEL chains directly.

    Parameters
    ----------
    template:
        A ``ChatPromptTemplate`` (e.g. from ``build_chat_prompt``).
    **kwargs:
        Variable values matching the template's input variables.

    Returns
    -------
    list[ChatMessage]
        Domain messages suitable for ``LLMProviderProtocol.chat()``.

    Raises
    ------
    KeyError
        If a required template variable is not provided in ``kwargs``.
    ValueError
        If the template produces a message role we cannot map to ``MessageRole``.

    Example
    -------
    >>> prompt = build_chat_prompt("You are a tutor.", "Explain {concept}.")
    >>> messages = format_to_chat_messages(prompt, concept="cosine similarity")
    >>> provider.chat(messages)
    """
    lc_messages: list[BaseMessage] = template.format_messages(**kwargs)
    return [
        ChatMessage(role=_lc_type_to_role(msg.type), content=str(msg.content))
        for msg in lc_messages
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LC_TYPE_TO_ROLE: dict[str, MessageRole] = {
    "system": MessageRole.system,
    "human": MessageRole.user,
    "ai": MessageRole.assistant,
}


def _lc_type_to_role(msg_type: str) -> MessageRole:
    """
    Convert a LangChain message type string to our ``MessageRole``.

    Raises
    ------
    ValueError
        If the type is not in the known mapping (e.g. ``"function"``,
        ``"tool"`` — those require a dedicated code path).
    """
    role = _LC_TYPE_TO_ROLE.get(msg_type)
    if role is None:
        raise ValueError(
            f"Cannot convert LangChain message type '{msg_type}' to MessageRole. "
            f"Supported types: {sorted(_LC_TYPE_TO_ROLE)}."
        )
    return role
