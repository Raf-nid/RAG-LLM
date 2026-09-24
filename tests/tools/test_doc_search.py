"""
Unit tests for DocSearchTool.

The tool is a placeholder — tests verify:
1. Schema validation (DocSearchInput)
2. Tool metadata (name, description, schema)
3. Execution returns a string (content format may change when Qdrant is added)
4. Unknown source filter returns an error string (not ToolError — tool is lenient here)
5. Source filter included in placeholder response
"""

import pytest

from rag_assistant.tools.doc_search import SUPPORTED_SOURCES, DocSearchInput, DocSearchTool


class TestDocSearchInput:
    def test_valid_minimal(self) -> None:
        inp = DocSearchInput(query="What is RAG?")
        assert inp.query == "What is RAG?"
        assert inp.source is None
        assert inp.top_k == 5

    def test_valid_with_source(self) -> None:
        inp = DocSearchInput(query="LangChain runnable", source="langchain")
        assert inp.source == "langchain"

    def test_valid_top_k_override(self) -> None:
        inp = DocSearchInput(query="q", top_k=10)
        assert inp.top_k == 10

    def test_empty_query_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DocSearchInput(query="")

    def test_missing_query_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DocSearchInput()  # type: ignore[call-arg]

    def test_top_k_below_minimum_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DocSearchInput(query="q", top_k=0)

    def test_top_k_above_maximum_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DocSearchInput(query="q", top_k=21)


class TestDocSearchToolExecute:
    def setup_method(self) -> None:
        self.tool = DocSearchTool()

    def _args(self, query: str, source: str | None = None, top_k: int = 5) -> DocSearchInput:
        return DocSearchInput(query=query, source=source, top_k=top_k)

    def test_returns_string(self) -> None:
        result = self.tool.execute(self._args("What is RAG?"))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_query_in_response(self) -> None:
        result = self.tool.execute(self._args("LangChain runnable"))
        assert "LangChain runnable" in result

    def test_valid_source_included_in_response(self) -> None:
        result = self.tool.execute(self._args("chains", source="langchain"))
        assert "langchain" in result

    def test_unknown_source_returns_error_string(self) -> None:
        result = self.tool.execute(self._args("q", source="wikipedia"))
        assert "Unknown source" in result
        assert "wikipedia" in result

    def test_top_k_in_response(self) -> None:
        result = self.tool.execute(self._args("q", top_k=3))
        assert "3" in result


class TestDocSearchToolMetadata:
    def setup_method(self) -> None:
        self.tool = DocSearchTool()

    def test_name(self) -> None:
        assert self.tool.name == "doc_search"

    def test_description_non_empty(self) -> None:
        assert len(self.tool.description) > 20

    def test_schema_has_query_field(self) -> None:
        schema = self.tool.to_tool_schema_params()
        assert "query" in schema.get("properties", {})

    def test_schema_has_source_field(self) -> None:
        schema = self.tool.to_tool_schema_params()
        assert "source" in schema.get("properties", {})

    def test_schema_has_top_k_field(self) -> None:
        schema = self.tool.to_tool_schema_params()
        assert "top_k" in schema.get("properties", {})

    def test_supported_sources_non_empty(self) -> None:
        assert len(SUPPORTED_SOURCES) > 0

    def test_repr(self) -> None:
        assert "doc_search" in repr(self.tool)
