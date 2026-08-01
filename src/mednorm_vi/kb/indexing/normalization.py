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


def tokens(text: str, *, strip_accents: bool = True) -> tuple[str, ...]:
    """Word tokens. Accent-stripped by default, which is the pre-Audit-0069 behaviour.

    ``strip_accents=False`` produces the accent-preserving channel that Audit 0069 needs:
    with accents collapsed, `sởi` (measles) and `sỏi` (calculus) are the same token, so an
    accent-sensitive tier cannot be built from the default form.
    """
    return tuple(_TOKEN.findall(normalize_text(text, strip_accents=strip_accents)))


def char_ngrams(text: str, *, n: int = 3, strip_accents: bool = True) -> tuple[str, ...]:
    value = f"  {normalize_text(text, strip_accents=strip_accents)}  "
    if len(value) <= n:
        return (value.strip(),) if value.strip() else ()
    return tuple(sorted({value[i : i + n] for i in range(len(value) - n + 1)}))


def _drop_accents(value: str) -> str:
    """Accent-strip without touching whitespace.

    `normalize_text` also collapses and strips spaces, so comparing a padded n-gram against
    it reports a difference for `"  s"` that has nothing to do with accents - which let every
    boundary gram into the accent-sensitive channel and collapsed it into a match-anything.
    """
    return "".join(
        ch for ch in unicodedata.normalize("NFD", value) if unicodedata.category(ch) != "Mn"
    )


def accent_marked_tokens(text: str) -> tuple[str, ...]:
    """Tokens that actually carry diacritic information.

    A token whose accented and accent-stripped forms are identical (`type`, `covid`) carries
    no accent evidence, so indexing it in the accent-sensitive channel would let anything
    match there. Keeping only genuinely accented tokens is what makes that channel able to
    tell `sởi` from `sỏi`.
    """
    return tuple(t for t in tokens(text, strip_accents=False) if t != _drop_accents(t))


def accent_marked_ngrams(text: str, *, n: int = 3) -> tuple[str, ...]:
    """Character n-grams that carry diacritic information; see `accent_marked_tokens`.

    Without this filter the padded boundary grams (`"  s"`, `"i  "`) are accent-free and
    shared by every string with the same first and last letter, which collapses the
    accent-sensitive tier into a match-anything channel.
    """
    return tuple(
        g for g in char_ngrams(text, n=n, strip_accents=False)
        if g != _drop_accents(g)
    )


__all__ = [
    "accent_marked_ngrams",
    "accent_marked_tokens",
    "char_ngrams",
    "normalize_text",
    "tokens",
]
