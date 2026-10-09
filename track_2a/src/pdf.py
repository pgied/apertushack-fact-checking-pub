"""Read the text of a PDF, page by page."""

from pathlib import Path

import pypdfium2 as pdfium


def pages(path: Path) -> list[str]:
    """The text of every page: page n, counted from 1 as PDF viewers do, is pages(path)[n - 1]."""
    pdf = pdfium.PdfDocument(path)
    try:
        return [clean(page.get_textpage().get_text_range()) for page in pdf]
    finally:
        pdf.close()


def clean(text: str) -> str:
    """pdfium ends lines with \\r\\n and marks the soft hyphen of a word broken across lines with \\ufffe."""
    return text.replace("\r\n", "\n").replace("￾", "")
