"""
Prompt templates for grounded RAG generation.

This module contains the prompt engineering for the RAG generation step.
The prompts are designed to:

1. Instruct the model to ground its answers in the provided context
2. Require explicit citations using chunk identifiers
3. Handle insufficient context gracefully (no hallucination)

Prompt Design Principles
------------------------
**Grounding**: The model must ONLY use information from the provided context.
Any answer not supported by the context should be declined.

**Citations**: Every factual claim must reference the source chunk(s).
We use a simple [1], [2] format that maps to chunk indices.

**Insufficient Context**: When the context doesn't contain relevant info,
the model should explicitly say so rather than making up an answer.

**Structured Output**: The response must be valid JSON matching our schema
so we can reliably extract the answer and citations.
"""

from dataclasses import dataclass

from rag_assistant.schemas.chunk import RetrievedChunk


@dataclass
class FormattedContext:
    """
    Context formatted for inclusion in the prompt.

    Attributes
    ----------
    text
        The formatted context text with chunk markers.
    chunk_mapping
        Mapping from citation index (1, 2, ...) to chunk_id.
    """

    text: str
    chunk_mapping: dict[int, str]


class RAGPromptBuilder:
    """
    Builds prompts for grounded RAG generation.

    The builder formats retrieved chunks into a context block with
    numbered markers, and creates system/user prompts that instruct
    the model to cite sources.
    """

    SYSTEM_PROMPT = (
        "You are a helpful technical assistant that answers questions "
        "based ONLY on the provided documentation context.\n\n"
        "CRITICAL RULES:\n"
        "1. ONLY use information from the CONTEXT below. Do not use prior knowledge.\n"
        "2. If the context does not contain relevant information, "
        "respond with insufficient_context=true.\n"
        "3. EVERY factual statement must include a citation like [1], [2], etc.\n"
        "4. Citations refer to the numbered sources in the context.\n"
        "5. Be concise but complete.\n\n"
        "You must respond with a JSON object matching this exact schema:\n"
        "{\n"
        '  "answer": "Your grounded answer with citations like [1], [2]...",\n'
        '  "citation_indices": [1, 2, ...],\n'
        '  "insufficient_context": false,\n'
        '  "confidence": "high" | "medium" | "low"\n'
        "}\n\n"
        "If you cannot answer from the context:\n"
        "{\n"
        '  "answer": "",\n'
        '  "citation_indices": [],\n'
        '  "insufficient_context": true,\n'
        '  "confidence": "low"\n'
        "}"
    )

    USER_TEMPLATE = (
        "CONTEXT:\n"
        "{context}\n\n"
        "QUESTION: {question}\n\n"
        "Remember: Answer ONLY from the context above. Cite sources with [1], [2], etc. "
        "If the context doesn't help, set insufficient_context to true."
    )

    def format_context(self, chunks: list[RetrievedChunk]) -> FormattedContext:
        """
        Format retrieved chunks into a numbered context block.

        Each chunk is prefixed with [N] where N is the 1-based index.
        This allows the model to cite specific chunks in its answer.

        Parameters
        ----------
        chunks
            Retrieved chunks ordered by relevance.

        Returns
        -------
        FormattedContext
            Formatted text and mapping from indices to chunk IDs.
        """
        if not chunks:
            return FormattedContext(text="No relevant documents found.", chunk_mapping={})

        lines: list[str] = []
        mapping: dict[int, str] = {}

        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.chunk_metadata
            header = f"[{i}] Source: {meta.source} | {meta.title}"
            if meta.section:
                header += f" | Section: {meta.section}"

            lines.append(header)
            lines.append(chunk.text)
            lines.append("")  # Blank line between chunks

            mapping[i] = meta.chunk_id

        return FormattedContext(
            text="\n".join(lines).strip(),
            chunk_mapping=mapping,
        )

    def build_system_prompt(self) -> str:
        """
        Get the system prompt for grounded generation.

        Returns
        -------
        str
            System prompt instructing the model on grounding and citations.
        """
        return self.SYSTEM_PROMPT

    def build_user_prompt(self, question: str, context: str) -> str:
        """
        Build the user prompt with question and context.

        Parameters
        ----------
        question
            The user's question.
        context
            Formatted context from retrieved chunks.

        Returns
        -------
        str
            Complete user prompt.
        """
        return self.USER_TEMPLATE.format(context=context, question=question)

    def build_messages(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[dict[str, str]], dict[int, str]]:
        """
        Build complete message list for the LLM.

        Parameters
        ----------
        question
            The user's question.
        chunks
            Retrieved chunks to use as context.

        Returns
        -------
        tuple[list[dict[str, str]], dict[int, str]]
            Tuple of (messages, chunk_mapping).
            Messages are dicts with 'role' and 'content'.
            chunk_mapping maps citation indices to chunk IDs.
        """
        formatted = self.format_context(chunks)

        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(question, formatted.text)},
        ]

        return messages, formatted.chunk_mapping
