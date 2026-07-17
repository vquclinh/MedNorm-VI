"""CLI for deterministic ICD-10 Vietnamese PDF conversion."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .normalization import normalize_rows
from .pdf_inspection import inspect_pdf
from .reporting import write_artifacts
from .row_parser import parse_pages
from .text_extraction import extract_pages
from .validation import validate_rows


def convert(pdf_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Convert the embedded text layer of an ICD PDF into ignored artifacts."""
    inspection = inspect_pdf(pdf_path)
    pages = extract_pages(pdf_path)
    parsed = parse_pages(pages)
    records = normalize_rows(parsed, source_document_sha256=inspection.sha256)
    validation = validate_rows(records)
    report = write_artifacts(
        output_dir,
        inspection=inspection,
        records=records,
        validation=validation,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_convert = sub.add_parser("convert", help="convert official ICD PDF text layer")
    p_convert.add_argument("--pdf", required=True)
    p_convert.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "convert":
        report = convert(args.pdf, args.output_dir)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        validation = report.get("validation", {})
        ok = bool(validation.get("ok", False)) if isinstance(validation, dict) else False
        return 0 if ok else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
