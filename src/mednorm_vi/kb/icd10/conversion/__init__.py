"""Reproducible ICD-10 Vietnamese PDF conversion pipeline.

The converter preserves official PDFs byte-for-byte. It reads the embedded text
layer with Poppler, reconstructs conservative ICD rows, and writes ignored
derived artifacts plus deterministic provenance.
"""

from .pdf_inspection import PdfInspection, inspect_pdf
from .row_parser import ParsedIcdRow, parse_pages
from .validation import ConversionValidation, validate_rows

__all__ = [
    "ConversionValidation",
    "ParsedIcdRow",
    "PdfInspection",
    "inspect_pdf",
    "parse_pages",
    "validate_rows",
]
