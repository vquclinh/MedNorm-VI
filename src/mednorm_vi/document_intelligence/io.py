"""Exact UTF-8 document loading for L1.

Loads a clinical document WITHOUT any silent normalization: newlines, repeated
whitespace, tabs, blank lines, punctuation, combining marks, duplicated
substrings, and leading/trailing whitespace are all preserved verbatim. The
decoded ``original_text`` is immutable and is the sole source of output offsets.

BOM policy is explicit (``preserve`` | ``strip`` | ``error``); stripping happens
only at load time (before offsets are established), never as a hidden later shift.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .unicode_utils import BOM, detect_newline_style, has_bom


class DocumentLoadError(Exception):
    """Raised on unreadable input, invalid UTF-8, or a policy violation."""


@dataclass(frozen=True, slots=True)
class DocumentSource:
    """Provenance/metadata for a loaded document (no text transformation)."""

    source_path: str | None
    encoding: str
    byte_length: int
    char_length: int
    sha256: str
    newline_style: str  # LF | CRLF | MIXED | CR | NONE
    has_bom: bool          # whether a BOM remains in original_text
    bom_policy: str
    bom_stripped: bool = False  # explicit metadata: a BOM was removed at load time


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """The immutable decoded text plus its source metadata and load warnings."""

    original_text: str
    source: DocumentSource
    warnings: tuple[str, ...] = ()


def _decode(raw: bytes, encoding: str, source_path: str | None) -> str:
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as exc:
        where = f" ({source_path})" if source_path else ""
        raise DocumentLoadError(
            f"invalid {encoding} input{where}: {exc}. L1 never silently repairs "
            "decoding; fix the source or the declared encoding."
        ) from exc


def load_text(
    raw: bytes,
    *,
    encoding: str = "utf-8",
    bom_policy: str = "preserve",
    source_path: str | None = None,
) -> LoadedDocument:
    """Decode raw bytes into a :class:`LoadedDocument` under an explicit policy."""
    if bom_policy not in {"preserve", "strip", "error"}:
        raise DocumentLoadError(f"unknown bom_policy {bom_policy!r}")
    byte_length = len(raw)
    decoded = _decode(raw, encoding, source_path)

    bom_present = has_bom(decoded)
    warnings: list[str] = []
    if bom_present and bom_policy == "error":
        where = f" ({source_path})" if source_path else ""
        raise DocumentLoadError(f"byte-order mark present but bom_policy=error{where}")
    if bom_present and bom_policy == "strip":
        # Strip only at load time, before any offset is established. This is an
        # explicit opt-in mode; record metadata + a structured warning so no
        # offset shift is ever invisible.
        decoded = decoded[len(BOM):]
        bom_present_final = False
        bom_stripped = True
        warnings.append(
            "bom.stripped: a leading BOM was removed under bom_policy=strip; the "
            "working original_text begins immediately after the BOM"
        )
    else:
        bom_present_final = bom_present
        bom_stripped = False

    sha256 = hashlib.sha256(decoded.encode("utf-8")).hexdigest()
    source = DocumentSource(
        source_path=source_path,
        encoding=encoding,
        byte_length=byte_length,
        char_length=len(decoded),
        sha256=sha256,
        newline_style=detect_newline_style(decoded),
        has_bom=bom_present_final,
        bom_policy=bom_policy,
        bom_stripped=bom_stripped,
    )
    return LoadedDocument(original_text=decoded, source=source, warnings=tuple(warnings))


def load_document(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    bom_policy: str = "preserve",
) -> LoadedDocument:
    """Read a file from disk and decode it exactly (no newline translation)."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"cannot read {p}: {exc}") from exc
    return load_text(raw, encoding=encoding, bom_policy=bom_policy, source_path=str(p))


def from_text(text: str, *, source_path: str | None = None) -> LoadedDocument:
    """Wrap an in-memory string as a :class:`LoadedDocument` (text kept verbatim)."""
    return load_text(text.encode("utf-8"), source_path=source_path)


__all__ = [
    "DocumentLoadError",
    "DocumentSource",
    "LoadedDocument",
    "load_text",
    "load_document",
    "from_text",
]
