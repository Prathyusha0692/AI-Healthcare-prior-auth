"""
PDF text extraction utilities using pdfplumber (primary) with pypdf fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger


def extract_text_from_pdf(file_path: str | Path) -> str:
    """
    Extract all text from a PDF file.

    Tries pdfplumber first (better table handling), falls back to pypdf.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # ── Try pdfplumber ────────────────────────────────────────────────────────
    try:
        import pdfplumber

        pages_text = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(f"[Page {i + 1}]\n{text}")
        if pages_text:
            logger.info(f"pdfplumber extracted {len(pages_text)} pages from {path.name}")
            return "\n\n".join(pages_text)
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e} — trying pypdf")

    # ── Fallback: pypdf ───────────────────────────────────────────────────────
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(f"[Page {i + 1}]\n{text}")
        logger.info(f"pypdf extracted {len(pages_text)} pages from {path.name}")
        return "\n\n".join(pages_text)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed for {path.name}: {e}") from e


def extract_text_from_bytes(content: bytes, filename: str = "document.pdf") -> str:
    """
    Extract text from PDF bytes (e.g., from an HTTP upload).

    Writes to a temp file, extracts, then cleans up.
    """
    import tempfile
    import os

    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        return extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)


def extract_pages_from_bytes(content: bytes, filename: str = "document.pdf") -> list[dict]:
    """
    Extract text page-by-page from PDF bytes.

    Returns:
        List of dicts: [{"page": 1, "text": "..."}, ...]
    """
    import tempfile
    import os

    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pages = []
        try:
            import pdfplumber
            with pdfplumber.open(tmp_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    pages.append({"page": i + 1, "text": text.strip()})
            return pages
        except Exception:
            from pypdf import PdfReader
            reader = PdfReader(tmp_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append({"page": i + 1, "text": text.strip()})
            return pages
    finally:
        os.unlink(tmp_path)
