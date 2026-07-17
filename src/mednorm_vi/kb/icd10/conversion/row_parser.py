"""Conservative row parser for the official Vietnamese ICD-10 PDF text layer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text_extraction import ExtractedPage

_ICD_CODE = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?$")
_CODE_ANYWHERE = re.compile(r"\b([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\b")
_ROW_START = re.compile(r"^\s*(?P<source_row>[0-9]{1,6})\s+")
_CHAPTER = re.compile(r"\b([IVXLCDM]{1,7}|XXII|XXI|XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X)\b")


@dataclass(frozen=True, slots=True)
class ParsedIcdRow:
    code_supplied: str
    label_vi: str
    label_en: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    chapter: str = ""
    block: str = ""
    parent: str = ""
    source_page: int = 0
    source_row: int = 0
    flags: tuple[str, ...] = field(default_factory=tuple)


def _looks_like_code(token: str) -> bool:
    return _ICD_CODE.match(token) is not None


def _pick_main_code(codes: list[str]) -> str:
    # Rows contain chapter/block/category codes before the final ICD concept.
    # Prefer the most specific repeated concept near the right side.
    for code in reversed(codes):
        if "." in code:
            return code
    return codes[-1] if codes else ""


def _block(codes: list[str]) -> str:
    for left, right in zip(codes, codes[1:], strict=False):
        if left != right and len(left) == 3 and len(right) == 3:
            return f"{left}-{right}"
    return ""


def _chapter(line: str) -> str:
    match = _CHAPTER.search(line[:80])
    return match.group(1) if match else ""


def _label_from_line(line: str, code: str) -> str:
    del code
    # The PDF table is wide and ends with one or more repeated code columns.
    # The most reliable label signal in the embedded text is the rightmost chunk
    # containing Vietnamese characters, before the trailing code-only columns.
    chunks = [chunk.strip() for chunk in re.split(r"\s{2,}", line) if chunk.strip()]
    vietnamese = [c for c in chunks if any(ord(ch) > 127 for ch in c)]
    if vietnamese:
        return vietnamese[-1]
    return ""


def parse_pages(pages: tuple[ExtractedPage, ...]) -> tuple[ParsedIcdRow, ...]:
    """Parse ICD rows from extracted PDF pages.

    The parser accepts only lines with a numeric official row id and at least
    one ICD-like code. Continuation lines are not merged into labels; that
    limitation is recorded by the conversion report and validation warnings.
    """
    rows: list[ParsedIcdRow] = []
    seen: set[tuple[int, int, str]] = set()
    for page in pages:
        for _line_no, line in enumerate(page.text.splitlines(), 1):
            start = _ROW_START.match(line)
            if start is None:
                continue
            codes = _CODE_ANYWHERE.findall(line)
            if not codes:
                continue
            main_code = _pick_main_code(codes)
            if not main_code:
                continue
            source_row = int(start.group("source_row"))
            key = (page.page_number, source_row, main_code)
            if key in seen:
                continue
            seen.add(key)
            label = _label_from_line(line, main_code)
            flags: list[str] = []
            if not label:
                flags.append("label_not_reconstructed")
            rows.append(
                ParsedIcdRow(
                    code_supplied=main_code,
                    label_vi=label,
                    chapter=_chapter(line),
                    block=_block(codes),
                    source_page=page.page_number,
                    source_row=source_row,
                    flags=tuple(flags),
                )
            )
    return tuple(rows)
