"""Embedded-text extraction for ICD-10 PDF conversion."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class TextExtractionError(RuntimeError):
    """Raised when Poppler text extraction fails."""


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    page_number: int
    text: str


def extract_pages(pdf_path: str | Path) -> tuple[ExtractedPage, ...]:
    """Extract all embedded text pages using ``pdftotext -layout``.

    No OCR or source-PDF mutation is performed.
    """
    if shutil.which("pdftotext") is None:
        raise TextExtractionError("pdftotext is required for ICD conversion")
    proc = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise TextExtractionError(proc.stderr.strip() or "pdftotext failed")
    raw_pages = proc.stdout.split("\f")
    pages = [
        ExtractedPage(page_number=i + 1, text=text)
        for i, text in enumerate(raw_pages)
        if text.strip()
    ]
    return tuple(pages)
