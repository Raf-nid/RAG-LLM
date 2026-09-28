"""Tests for PDF loader."""

import tempfile
from pathlib import Path

import fitz
import pytest

from rag_assistant.rag.ingestion.pdf_loader import (
    PDFDocument,
    PDFPage,
    load_pdf,
    pdf_to_text_with_tables,
)


@pytest.fixture
def sample_pdf() -> Path:
    """Create a simple test PDF."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        doc = fitz.open()
        page = doc.new_page()

        # Add some text
        text_point = fitz.Point(72, 72)
        page.insert_text(text_point, "Hello World\nThis is a test document.")

        # Add more text on next line
        text_point = fitz.Point(72, 120)
        page.insert_text(text_point, "Page 1 content with numbers: 12345")

        # Add second page
        page2 = doc.new_page()
        text_point = fitz.Point(72, 72)
        page2.insert_text(text_point, "Second page content.")

        doc.save(f.name)
        doc.close()

        return Path(f.name)


@pytest.fixture
def pdf_with_metadata() -> Path:
    """Create a PDF with metadata."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        doc = fitz.open()
        doc.set_metadata(
            {
                "title": "Test Report 2023",
                "author": "Test Author",
            }
        )

        page = doc.new_page()
        text_point = fitz.Point(72, 72)
        page.insert_text(text_point, "Document with metadata")

        doc.save(f.name)
        doc.close()

        return Path(f.name)


class TestLoadPDF:
    """Tests for load_pdf function."""

    def test_load_basic_pdf(self, sample_pdf: Path) -> None:
        doc = load_pdf(sample_pdf)

        assert isinstance(doc, PDFDocument)
        assert doc.total_pages == 2
        assert len(doc.pages) == 2

    def test_extracts_text(self, sample_pdf: Path) -> None:
        doc = load_pdf(sample_pdf)

        # First page should have our text
        assert "Hello World" in doc.pages[0].text
        assert "test document" in doc.pages[0].text

        # Second page
        assert "Second page" in doc.pages[1].text

    def test_page_numbers_are_one_indexed(self, sample_pdf: Path) -> None:
        doc = load_pdf(sample_pdf)

        assert doc.pages[0].page_number == 1
        assert doc.pages[1].page_number == 2

    def test_extracts_metadata(self, pdf_with_metadata: Path) -> None:
        doc = load_pdf(pdf_with_metadata)

        assert doc.title == "Test Report 2023"
        assert doc.author == "Test Author"

    def test_uses_filename_when_no_title(self, sample_pdf: Path) -> None:
        doc = load_pdf(sample_pdf)

        # Should use stem of filename
        assert doc.title == sample_pdf.stem

    def test_full_text_property(self, sample_pdf: Path) -> None:
        doc = load_pdf(sample_pdf)

        full_text = doc.full_text
        assert "Hello World" in full_text
        assert "Second page" in full_text

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_pdf("/nonexistent/file.pdf")

    def test_stores_path(self, sample_pdf: Path) -> None:
        doc = load_pdf(sample_pdf)
        assert doc.path == str(sample_pdf)


class TestPDFPage:
    """Tests for PDFPage dataclass."""

    def test_page_dataclass(self) -> None:
        page = PDFPage(
            page_number=1,
            text="Some content",
            tables=["| col1 | col2 |"],
        )

        assert page.page_number == 1
        assert page.text == "Some content"
        assert len(page.tables) == 1


class TestPDFToTextWithTables:
    """Tests for pdf_to_text_with_tables function."""

    def test_combines_pages(self, sample_pdf: Path) -> None:
        text = pdf_to_text_with_tables(sample_pdf)

        assert "Page 1" in text
        assert "Page 2" in text
        assert "Hello World" in text
        assert "Second page" in text

    def test_includes_page_markers(self, sample_pdf: Path) -> None:
        text = pdf_to_text_with_tables(sample_pdf)

        assert "--- Page 1 ---" in text
        assert "--- Page 2 ---" in text
