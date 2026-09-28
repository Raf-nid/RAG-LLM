"""
PDF document loader using PyMuPDF.

This module provides utilities to extract text and tables from PDF documents.
PyMuPDF (fitz) is a well-maintained library with good performance.

Key Features
------------
- Text extraction with layout preservation
- Table detection and extraction
- Page-by-page processing
- Metadata extraction (title, author, etc.)

Limitations
-----------
- Table extraction quality depends on PDF structure
- Scanned PDFs require OCR (not included here)
- Complex layouts may lose some structure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # type: ignore[import-untyped]  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PDFPage:
    """
    Extracted content from a single PDF page.

    Attributes
    ----------
    page_number
        1-indexed page number.
    text
        Extracted text content.
    tables
        List of tables found on this page (as markdown).
    """

    page_number: int
    text: str
    tables: list[str]


@dataclass
class PDFDocument:
    """
    Extracted content from a PDF document.

    Attributes
    ----------
    path
        Path to the source PDF.
    title
        Document title (from metadata or filename).
    author
        Document author (if available).
    pages
        List of extracted pages.
    total_pages
        Total number of pages.
    """

    path: str
    title: str
    author: str | None
    pages: list[PDFPage]
    total_pages: int

    @property
    def full_text(self) -> str:
        """Concatenate all page texts."""
        return "\n\n".join(page.text for page in self.pages if page.text)


def extract_tables_from_page(page: fitz.Page) -> list[str]:
    """
    Extract tables from a PDF page as markdown.

    Uses PyMuPDF's table detection. Results vary by PDF quality.

    Parameters
    ----------
    page
        PyMuPDF page object.

    Returns
    -------
    list[str]
        Tables formatted as markdown.
    """
    tables: list[str] = []

    try:
        # PyMuPDF 1.23+ has find_tables()
        table_finder = page.find_tables()
        for table in table_finder.tables:
            # Convert to pandas and then to markdown
            df = table.to_pandas()
            if not df.empty:
                md_table = df.to_markdown(index=False)
                if md_table:
                    tables.append(md_table)
    except Exception as e:
        logger.debug("Table extraction failed on page %d: %s", page.number + 1, e)

    return tables


def load_pdf(path: str | Path) -> PDFDocument:
    """
    Load and extract content from a PDF file.

    Parameters
    ----------
    path
        Path to the PDF file.

    Returns
    -------
    PDFDocument
        Extracted document with text and tables.

    Raises
    ------
    FileNotFoundError
        If the PDF file doesn't exist.
    fitz.FileDataError
        If the file is not a valid PDF.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(path))

    # Extract metadata
    metadata = doc.metadata
    title = metadata.get("title") or path.stem
    author = metadata.get("author") or None

    pages: list[PDFPage] = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # Extract text
        text = page.get_text("text")

        # Extract tables
        tables = extract_tables_from_page(page)

        pages.append(
            PDFPage(
                page_number=page_idx + 1,  # 1-indexed
                text=text.strip(),
                tables=tables,
            )
        )

    doc.close()

    return PDFDocument(
        path=str(path),
        title=title,
        author=author,
        pages=pages,
        total_pages=len(pages),
    )


def pdf_to_text_with_tables(path: str | Path) -> str:
    """
    Extract text from PDF with tables inline.

    Convenience function that returns a single string with
    text and tables interleaved per page.

    Parameters
    ----------
    path
        Path to the PDF file.

    Returns
    -------
    str
        Combined text content with tables.
    """
    doc = load_pdf(path)

    parts: list[str] = []
    for page in doc.pages:
        if page.text:
            parts.append(f"--- Page {page.page_number} ---")
            parts.append(page.text)
            for i, table in enumerate(page.tables, 1):
                parts.append(f"\n[Table {i}]\n{table}")

    return "\n\n".join(parts)
