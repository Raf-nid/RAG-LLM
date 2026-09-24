"""
Schema: QuestionAnalysis.

Defines the structured output produced when the LLM analyses an incoming
technical question.  This schema is provider-agnostic: it is a plain Pydantic
model with no dependency on any LLM SDK.

Usage in the pipeline
---------------------
When a user submits a question, the assistant uses this schema to understand:

- What *kind* of question it is (so it can route or frame the retrieval).
- How *difficult* the question is (so it can calibrate the answer depth).
- Which *concepts* are involved (so it can highlight learning material).
- A concise *answer* grounded in retrieved context (in later stages, the
  RAG pipeline will populate this from retrieved documents; here it is
  generated directly by the LLM).

Extending this schema
---------------------
Add fields here when the pipeline needs more structured information.
Always keep fields that the LLM is realistically able to fill correctly.
Overly complex schemas increase the probability of validation failures.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class QuestionIntent(StrEnum):
    """The intent or type of a technical question."""

    conceptual = "conceptual"  # "What is X?" / "How does X work?"
    practical = "practical"  # "How do I do X?" / "Show me code for X"
    comparison = "comparison"  # "What is the difference between X and Y?"
    debug = "debug"  # "Why does X not work?" / "What is wrong with my code?"
    other = "other"  # Does not fit the above categories


class Difficulty(StrEnum):
    """Estimated difficulty of the question for someone learning AI Engineering."""

    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class QuestionAnalysis(BaseModel):
    """
    Structured analysis of a technical question.

    All fields are required.  The LLM must populate all of them.
    Extra fields returned by the LLM are silently ignored.

    Fields
    ------
    intent
        The category that best describes what the user wants to know.
    difficulty
        Estimated difficulty level for an AI Engineering learner.
    concepts
        List of key technical concepts the question touches on.
        Should contain at least one entry.  The LLM should be specific
        (e.g. "vector similarity" rather than "AI").
    answer
        A concise, accurate answer to the question.  In the full RAG
        pipeline this will be grounded in retrieved documents.
    """

    model_config = ConfigDict(extra="ignore")

    intent: QuestionIntent = Field(
        description="The category that best describes what the user is asking."
    )
    difficulty: Difficulty = Field(
        description="Estimated difficulty level for an AI Engineering learner."
    )
    concepts: list[str] = Field(
        description="Key technical concepts involved in the question.",
        min_length=1,
    )
    answer: str = Field(
        description="A concise, accurate answer to the question.",
        min_length=1,
    )
