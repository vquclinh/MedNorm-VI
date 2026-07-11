"""Tokenization strategies for WER (PROVISIONAL — organizer detail unpublished).

Three explicit strategies are provided; the provisional default lives in
``configs/evaluation/provisional_v1.yaml``. None of these normalizes the
organizer-facing entity text (no lower-casing, accent stripping, or Unicode
folding) unless the strategy name explicitly declares it. All are pure and
deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# A token is a word run (Unicode letters/digits/underscore) OR a single
# non-space, non-word character (punctuation). Vietnamese letters are word chars
# under the Unicode default of ``re``.
_WORD_OR_PUNCT = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_whitespace(text: str) -> tuple[str, ...]:
    """Split on runs of whitespace (repeated whitespace collapses). No normalization."""
    return tuple(text.split())


def tokenize_whitespace_punctuation(text: str) -> tuple[str, ...]:
    """Word tokens plus each punctuation mark as its own token. No normalization."""
    return tuple(_WORD_OR_PUNCT.findall(text))


def tokenize_character_diagnostic(text: str) -> tuple[str, ...]:
    """Every Unicode code point (including spaces) is a token. Diagnostic only."""
    return tuple(text)


#: Registry of tokenization strategies by name.
TOKENIZERS: dict[str, Callable[[str], tuple[str, ...]]] = {
    "whitespace": tokenize_whitespace,
    "whitespace-punctuation": tokenize_whitespace_punctuation,
    "character-diagnostic": tokenize_character_diagnostic,
}


def get_tokenizer(name: str) -> Callable[[str], tuple[str, ...]]:
    """Return the tokenizer callable for ``name`` or raise ``KeyError``."""
    if name not in TOKENIZERS:
        raise KeyError(f"unknown tokenization strategy {name!r}; choose from {sorted(TOKENIZERS)}")
    return TOKENIZERS[name]


__all__ = [
    "tokenize_whitespace",
    "tokenize_whitespace_punctuation",
    "tokenize_character_diagnostic",
    "TOKENIZERS",
    "get_tokenizer",
]
