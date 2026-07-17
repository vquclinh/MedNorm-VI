"""Text normalization for deterministic local KB retrieval."""

from __future__ import annotations

import re
import unicodedata

_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def normalize_text(text: str, *, strip_accents: bool = False) -> str:
    """Lowercase, normalize Unicode, optionally remove accents, and compact space."""
    value = unicodedata.normalize("NFKC", text).casefold()
    if strip_accents:
        value = "".join(
            ch for ch in unicodedata.normalize("NFD", value)
            if unicodedata.category(ch) != "Mn"
        )
    return _SPACE.sub(" ", value).strip()


def tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(normalize_text(text, strip_accents=True)))


def char_ngrams(text: str, *, n: int = 3) -> tuple[str, ...]:
    value = f"  {normalize_text(text, strip_accents=True)}  "
    if len(value) <= n:
        return (value.strip(),) if value.strip() else ()
    return tuple(sorted({value[i : i + n] for i in range(len(value) - n + 1)}))


__all__ = ["char_ngrams", "normalize_text", "tokens"]
