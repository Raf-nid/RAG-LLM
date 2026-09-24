"""
LCEL (LangChain Expression Language) chain builders.

What is LCEL?
-------------
LCEL is LangChain's composition mechanism. The ``|`` operator connects any
two ``Runnable`` objects into a ``RunnableSequence``:

    chain = prompt | model | parser

This is equivalent to:

    parser.invoke(model.invoke(prompt.invoke(inputs)))

but with four critical differences:

1. **Tracing** — each step is a separate span in LangSmith.
2. **Streaming** — ``chain.stream()`` propagates tokens through the chain.
3. **Batching** — ``chain.batch([...])`` runs multiple inputs efficiently.
4. **Async** — ``await chain.ainvoke(...)`` works natively.

When to use LCEL vs provider.chat()
------------------------------------
- Use ``provider.chat(messages)`` (our protocol) when you need domain types
  (``LLMResponse``, token counts, latency) and the application controls the
  exact messages sent.
- Use LCEL chains when composing multiple LangChain components, particularly
  templates + model + parsers in a RAG pipeline (where the retriever's output
  feeds directly into the prompt).

What this module provides
--------------------------
``build_chat_chain``
    A text-output chain: ``prompt | model | StrOutputParser``.
    Input: dict of template variables. Output: ``str``.

``build_structured_chain``
    A typed-output chain: ``prompt | model.with_structured_output(schema)``.
    Input: dict of template variables. Output: instance of ``schema``.

``call_structured_native``
    Direct invocation of ``model.with_structured_output(schema)`` on existing
    ``ChatMessage`` messages. This is the *alternative* to ``call_structured``
    from ``structured.py``:

    +-------------------------+-----------------------------------------+
    | ``call_structured``     | ``call_structured_native``              |
    +-------------------------+-----------------------------------------+
    | Any provider            | Requires LangChain BaseChatModel        |
    | Prompt injection + JSON | Provider-native JSON / function calling  |
    | Works on small models   | More reliable on supported models       |
    | Our domain types in/out | BaseChatModel in, typed Pydantic out    |
    +-------------------------+-----------------------------------------+

    Get the underlying model via ``provider.as_lc_model()``.

Provider isolation
------------------
These functions accept ``BaseChatModel``, not our ``LLMProviderProtocol``.
That is intentional: LCEL is a LangChain-level concern. Application code that
needs provider-neutral behaviour uses ``LLMProviderProtocol`` and
``call_structured``. Code that explicitly opts into LCEL features uses
``as_lc_model()`` and the functions here.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from ._langchain import to_langchain_messages
from .interface import ChatMessage

__all__ = [
    "acall_structured_native",
    "build_chat_chain",
    "build_structured_chain",
    "call_structured_native",
]


def build_chat_chain(
    prompt: ChatPromptTemplate,
    model: BaseChatModel,
) -> Runnable:
    """
    Build a string-output LCEL chain.

    Chain: ``prompt | model | StrOutputParser``

    The chain accepts a dict whose keys match the template's input variables
    and returns a plain string (the model's text response).

    Parameters
    ----------
    prompt:
        A ``ChatPromptTemplate``. Build one with ``build_chat_prompt()``.
    model:
        A LangChain chat model. Retrieve via ``provider.as_lc_model()``.

    Returns
    -------
    Runnable
        A composable runnable. Call ``chain.invoke({"var": "value"})``.

    Example
    -------
    >>> prompt = build_chat_prompt("You are a tutor.", "Explain {concept}.")
    >>> chain = build_chat_chain(prompt, provider.as_lc_model())
    >>> answer = chain.invoke({"concept": "cosine similarity"})
    """
    return prompt | model | StrOutputParser()


def build_structured_chain[T: BaseModel](
    prompt: ChatPromptTemplate,
    model: BaseChatModel,
    schema: type[T],
) -> Runnable:
    """
    Build a structured-output LCEL chain.

    Chain: ``prompt | model.with_structured_output(schema)``

    Uses the provider's native structured output mechanism:

    - **Groq**: function calling (JSON via the OpenAI tools API)
    - **Ollama**: JSON schema enforcement at the sampling level

    Both are more reliable than prompt-injection-based parsing for the same
    reason: the constraint is enforced at generation time, not after.

    Parameters
    ----------
    prompt:
        A ``ChatPromptTemplate``.
    model:
        A LangChain chat model. Retrieve via ``provider.as_lc_model()``.
    schema:
        The Pydantic model to extract. Must be a concrete ``BaseModel`` subclass.

    Returns
    -------
    Runnable
        Call ``chain.invoke({"var": "value"})`` to get an instance of ``schema``.

    Example
    -------
    >>> from rag_assistant.schemas.question_analysis import QuestionAnalysis
    >>> chain = build_structured_chain(prompt, provider.as_lc_model(), QuestionAnalysis)
    >>> analysis = chain.invoke({"question": "What is BM25?"})
    >>> assert isinstance(analysis, QuestionAnalysis)
    """
    return prompt | model.with_structured_output(schema)


def call_structured_native[T: BaseModel](
    model: BaseChatModel,
    messages: list[ChatMessage],
    schema: type[T],
) -> T:
    """
    Extract a typed Pydantic object using the provider's native mechanism.

    This is the single-call alternative to ``call_structured()`` when you
    have an explicit ``BaseChatModel`` (not just a ``LLMProviderProtocol``).
    It delegates structure enforcement to the provider rather than relying on
    prompt injection and text parsing.

    Use ``provider.as_lc_model()`` to get the underlying model.

    Parameters
    ----------
    model:
        The LangChain chat model (e.g. ``ChatGroq`` or ``ChatOllama``).
        Get it via ``your_provider.as_lc_model()``.
    messages:
        Conversation messages in domain type form.
    schema:
        Pydantic model class to extract.

    Returns
    -------
    T
        A validated instance of ``schema``.

    Raises
    ------
    TypeError
        If the model returns something other than an instance of ``schema``.
        This should not happen in normal operation but guards against
        unexpected LangChain parser behaviour.
    """
    lc_messages = to_langchain_messages(messages)
    structured_model = model.with_structured_output(schema)
    result = structured_model.invoke(lc_messages)
    if not isinstance(result, schema):
        raise TypeError(
            f"Expected {schema.__name__}, got {type(result).__name__}. "
            "This may indicate the model returned a partial or malformed response."
        )
    return result  # type: ignore[return-value]


async def acall_structured_native[T: BaseModel](
    model: BaseChatModel,
    messages: list[ChatMessage],
    schema: type[T],
) -> T:
    """
    Async variant of ``call_structured_native``.

    Identical semantics — uses ``ainvoke`` instead of ``invoke``.
    """
    lc_messages = to_langchain_messages(messages)
    structured_model = model.with_structured_output(schema)
    result = await structured_model.ainvoke(lc_messages)
    if not isinstance(result, schema):
        raise TypeError(f"Expected {schema.__name__}, got {type(result).__name__}.")
    return result  # type: ignore[return-value]
