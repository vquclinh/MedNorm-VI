"""Deterministic corpus discovery for the organizer's PUBLIC INPUT documents.

INPUT ONLY. These documents have no ground truth and must never be labeled,
committed, or treated as evaluable (see ``data/README.md``). This loader reads
them read-only, in a stable numeric-then-lexical order.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_NUM = re.compile(r"^(\d+)$")


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One public input document, read verbatim (never mutated)."""

    source_name: str  # file stem, e.g. "42"
    path: str
    text: str

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _sort_key(stem: str) -> tuple[int, int | str]:
    """Numeric stems sort numerically (1, 2, ... 100), others lexically after."""
    m = _NUM.match(stem)
    return (0, int(m.group(1))) if m else (1, stem)


def discover_documents(directory: str | Path, *, pattern: str = "*.txt") -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(root.glob(pattern), key=lambda p: _sort_key(p.stem))


def load_corpus(directory: str | Path, *, pattern: str = "*.txt",
                encoding: str = "utf-8") -> list[CorpusDocument]:
    """Load every document under ``directory`` deterministically (read-only)."""
    docs: list[CorpusDocument] = []
    for p in discover_documents(directory, pattern=pattern):
        docs.append(CorpusDocument(source_name=p.stem, path=str(p),
                                   text=p.read_text(encoding=encoding)))
    return docs


def corpus_sha256(documents: list[CorpusDocument]) -> str:
    """Content hash over the corpus (order-independent, content-defined)."""
    h = hashlib.sha256()
    for d in sorted(documents, key=lambda x: _sort_key(x.source_name)):
        h.update(f"{d.source_name}|{d.content_sha256}\n".encode())
    return h.hexdigest()


__all__ = ["CorpusDocument", "discover_documents", "load_corpus", "corpus_sha256"]
