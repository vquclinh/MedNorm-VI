"""Whole-phrase proposals from the governed KB (0081).

These are a **proposal source only**. An alias hit says "this phrase exists in the governed
vocabulary", which is evidence that a mention is here; it never emits an organizer entity by
itself and it never emits a code. Codes remain GraphCENT's decision.

Matching is token-aligned, not substring. The document and every alias are cut into word
tokens once, and an alias matches only when its whole token sequence lines up with a whole
token sequence in the document. That is what makes a mid-word hit impossible: a two-character
abbreviation inside an ordinary word is not a token, so it cannot match. Short single-token
aliases are held to a stricter rule again - they must match with their exact casing - because
a three-letter lowercase alias is otherwise a licence to annotate ordinary prose.

Nothing here is specific to any test set: the aliases come from whatever governed index is
passed in, and the rules are about token shape, not about particular words.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

#: A word token. `\w` under Unicode covers Vietnamese letters with diacritics, and digits
#: so that `500mg`-style tokens stay whole.
_TOKEN = re.compile(r"\w+", re.UNICODE)

#: A single-token alias at or below this length must match its exact casing. Short forms are
#: overwhelmingly acronyms, and a case-insensitive three-letter match fires on ordinary text.
SHORT_TOKEN_MAX = 4

#: A single-token alias shorter than this is never used at all: it carries no evidence.
MIN_ALIAS_CHARS = 3


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: int
    end: int

    @property
    def folded(self) -> str:
        return fold(self.text)


def fold(text: str) -> str:
    """Case-folded, NFC-normalized. Diacritics are preserved - they are meaning, not noise."""
    return unicodedata.normalize("NFC", text).casefold()


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]


@dataclass(frozen=True, slots=True)
class AliasEntry:
    """One governed surface form, reduced to the tokens that must line up."""

    phrase: str
    tokens: tuple[str, ...]
    entity_type: str
    case_sensitive: bool

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class AliasLexicon:
    """Governed surface forms keyed by their first token, for a single pass over a document."""

    by_first_token: dict[str, list[AliasEntry]] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)

    def add(self, phrase: str, entity_type: str) -> bool:
        tokens = tokenize(phrase)
        if not tokens:
            self._skip("no_tokens")
            return False
        folded = tuple(t.folded for t in tokens)
        if len(folded) == 1:
            if len(folded[0]) < MIN_ALIAS_CHARS:
                self._skip("single_token_too_short")
                return False
            case_sensitive = len(folded[0]) <= SHORT_TOKEN_MAX
        else:
            case_sensitive = False
        entry = AliasEntry(
            phrase=phrase, tokens=folded, entity_type=entity_type,
            case_sensitive=case_sensitive,
        )
        bucket = self.by_first_token.setdefault(folded[0], [])
        if entry not in bucket:
            bucket.append(entry)
        return True

    def _skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    @property
    def size(self) -> int:
        return sum(len(v) for v in self.by_first_token.values())

    def as_dict(self) -> dict[str, Any]:
        return {"entries": self.size, "skipped": dict(self.skipped)}


@dataclass(frozen=True, slots=True)
class AliasHit:
    start: int
    end: int
    text: str
    entity_type: str
    phrase: str


def find_alias_hits(source: str, lexicon: AliasLexicon) -> list[AliasHit]:
    """Every whole-token alias occurrence, longest match first at each position.

    Longest-first matters: when a governed vocabulary contains both a phrase and one of its
    words, the phrase is the complete concept and the single word is a fragment of it.
    """
    tokens = tokenize(source)
    folded = [t.folded for t in tokens]
    hits: list[AliasHit] = []
    index = 0
    while index < len(tokens):
        best: AliasEntry | None = None
        for entry in sorted(
            lexicon.by_first_token.get(folded[index], ()), key=lambda e: -e.length
        ):
            end_index = index + entry.length
            if end_index > len(tokens):
                continue
            if tuple(folded[index:end_index]) != entry.tokens:
                continue
            if entry.case_sensitive and not _same_case(tokens, index, entry):
                continue
            best = entry
            break
        if best is None:
            index += 1
            continue
        start, end = tokens[index].start, tokens[index + best.length - 1].end
        hits.append(
            AliasHit(
                start=start, end=end, text=source[start:end],
                entity_type=best.entity_type, phrase=best.phrase,
            )
        )
        index += best.length
    return hits


def _same_case(tokens: list[Token], index: int, entry: AliasEntry) -> bool:
    original = tokenize(entry.phrase)
    return all(
        tokens[index + offset].text == original[offset].text
        for offset in range(entry.length)
    )


def surface_forms(record: dict[str, Any]) -> Iterator[str]:
    """Canonical name and aliases of one governed record, whatever the index layout."""
    name = str(record.get("canonical_name") or "").strip()
    if name:
        yield name
    for alias in record.get("aliases") or ():
        text = str(alias).strip()
        if text:
            yield text


def build_lexicon(
    sources: Iterable[tuple[Any, str]], *, max_tokens: int = 8
) -> AliasLexicon:
    """`(index, organizer_type)` pairs -> one lexicon.

    Very long surface forms are dropped: a governed description of eight or more words is a
    definition, not something a clinician writes in a note, and matching one would only ever
    fire on a coincidence.
    """
    lexicon = AliasLexicon()
    for index, entity_type in sources:
        records = getattr(index, "records", {}) or {}
        for record in records.values():
            if not isinstance(record, dict):
                continue
            for phrase in surface_forms(record):
                if len(tokenize(phrase)) > max_tokens:
                    lexicon._skip("too_many_tokens")
                    continue
                lexicon.add(phrase, entity_type)
    return lexicon


__all__ = [
    "MIN_ALIAS_CHARS",
    "SHORT_TOKEN_MAX",
    "AliasEntry",
    "AliasHit",
    "AliasLexicon",
    "Token",
    "build_lexicon",
    "find_alias_hits",
    "fold",
    "surface_forms",
    "tokenize",
]
