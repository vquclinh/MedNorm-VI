"""Deterministic canonical L1 token view (NOT a model tokenizer).

Word/number tokens keep internal decimal separators, unit slashes, and hyphens
(e.g. ``14.43``, ``14,43``, ``325-650``, ``mg/ml``); each punctuation mark is its
own token. Model-specific tokenizers added later must map back through this view.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .unicode_utils import strip_accents

# Word/number runs with internal connectors, OR a single punctuation/symbol char.
_TOKEN = re.compile(r"[^\W_]+(?:[.,/\-][^\W_]+)*|[^\s\w]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class Token:
    start: int  # absolute
    end: int
    text: str
    normalized: str
    category: str  # "word" | "number" | "punct"
    is_punctuation: bool
    whitespace_before: bool


def _normalize_token(text: str) -> str:
    return strip_accents(unicodedata.normalize("NFC", text).casefold())


def tokenize(original_text: str, content_start: int, content_end: int) -> list[Token]:
    """Tokenize ``original_text[content_start:content_end]`` into absolute tokens."""
    segment = original_text[content_start:content_end]
    tokens: list[Token] = []
    for m in _TOKEN.finditer(segment):
        abs_start = content_start + m.start()
        abs_end = content_start + m.end()
        text = m.group()
        first = text[0]
        if first.isdigit():
            category = "number"
            is_punct = False
        elif first.isalpha():
            category = "word"
            is_punct = False
        else:
            category = "punct"
            is_punct = True
        ws_before = abs_start > 0 and original_text[abs_start - 1].isspace()
        tokens.append(Token(
            start=abs_start,
            end=abs_end,
            text=text,
            normalized=_normalize_token(text),
            category=category,
            is_punctuation=is_punct,
            whitespace_before=ws_before,
        ))
    return tokens


__all__ = ["Token", "tokenize"]
