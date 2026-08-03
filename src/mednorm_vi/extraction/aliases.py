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

**Alias quality is a governed-data question, not a word list.** The first real smoke produced
329 alias hits in three documents, and the worst of them were single ordinary Vietnamese words
like "other", "not" and "disease". Auditing the governed ICD index explains why: those are
`canonical_name` values, and one of them is the recorded name of twenty-five different
concepts. A surface form that names twenty-five concepts cannot identify which concept a
mention means - that is measurable from the KB itself, so this module measures it:

* **ambiguity** - a normalized form is counted across the concepts that carry it, and a form
  above `max_concepts_per_form` is refused. Nothing about which words they are;
* **governed quality metadata** - the ICD index already records `name_quality` and
  `quality_flags` (`suspect_pdf_fragment:leading_marker`, `suspect_trailing_function_word`,
  `suspect_guidance_text`). A label the KB itself marks as suspect is not used as an alias;
* **shape** - empty, punctuation-only and over-long forms are refused.

None of this is specific to any test set: the thresholds are about how a form behaves across
the governed vocabulary, not about particular words. And it is only the second line of
defence - the first is that an alias hit is a *proposal*, never an entity on its own.
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

#: How many governed concepts one normalized surface form may name and still identify a
#: mention. Measured from the KB: in the governed ICD index 9,940 forms name two or more
#: concepts and 647 name ten or more, all of them label fragments rather than clinical terms.
MAX_CONCEPTS_PER_FORM = 3

#: A single token appearing in more than this share of an ontology's labels is a common word
#: of that vocabulary rather than a term that identifies anything. Measured, not listed: in
#: the governed ICD index "disease" occurs in 16.3% of labels, "not" in 9.4% and "other" in
#: 8.8%, while a drug name like `aspirin` occurs in 0.5% of RxNorm labels.
MAX_SINGLE_TOKEN_LABEL_SHARE = 0.01

#: Governed label-quality metadata already present on the ICD records. A label the knowledge
#: base itself marks as suspect is not a surface form a clinician wrote.
QUALITY_CLEAN = "clean"
REJECT_AMBIGUOUS_FORM = "ambiguous_across_concepts"
REJECT_SUSPECT_LABEL = "governed_label_marked_suspect"
REJECT_COMMON_TOKEN = "single_token_is_a_common_word"


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
    #: Normalized form -> how many governed concepts carry it. Recorded so a hit can report
    #: how identifying it actually was.
    ambiguity: dict[str, int] = field(default_factory=dict)

    def concepts_named_by(self, phrase: str) -> int:
        return self.ambiguity.get(form_key(phrase), 1)

    def add(self, phrase: str, entity_type: str) -> bool:
        tokens = tokenize(phrase)
        if not tokens:
            self._skip("no_tokens")
            return False
        if not any(t.folded.strip() for t in tokens):
            self._skip("punctuation_only")
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
        return {
            "entries": self.size,
            "distinct_forms_measured": len(self.ambiguity),
            "skipped": dict(self.skipped),
        }


@dataclass(frozen=True, slots=True)
class AliasHit:
    start: int
    end: int
    text: str
    entity_type: str
    phrase: str
    #: How many governed concepts this surface form names. 1 is an identifying alias.
    concepts_named: int = 1


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
                concepts_named=lexicon.concepts_named_by(best.phrase),
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


def label_is_suspect(record: dict[str, Any]) -> bool:
    """True when the governed index itself marks this label as unreliable.

    The ICD index records `name_quality` and `quality_flags` from its own title repair. A
    label flagged there is a PDF fragment, a trailing function word or guidance prose, and
    using it as an alias would annotate the note with the knowledge base's own defects.
    """
    metadata = record.get("metadata") or {}
    if str(metadata.get("name_quality", QUALITY_CLEAN)) != QUALITY_CLEAN:
        return True
    return bool(str(metadata.get("quality_flags", "")).strip())


def form_key(phrase: str) -> str:
    """The key a form is counted under: its TOKEN SEQUENCE, which is what matching compares.

    Keying on the raw string instead would count `Bệnh:` and `bệnh` separately, and the
    punctuation-bearing variant would then look unique while matching exactly the same
    single token. That is how the first repair still let common words through.
    """
    return " ".join(t.folded for t in tokenize(phrase))


def form_ambiguity(sources: Iterable[tuple[Any, str]]) -> dict[str, int]:
    """Token-sequence key -> how many governed concepts carry it.

    This is the measurement the alias policy is built on. It is computed from the supplied
    indices, so it describes whatever knowledge base is actually in use.
    """
    counts: dict[str, set[str]] = {}
    for index, _ in sources:
        for concept_id, record in (getattr(index, "records", {}) or {}).items():
            if not isinstance(record, dict):
                continue
            for phrase in surface_forms(record):
                key = form_key(phrase)
                if key:
                    counts.setdefault(key, set()).add(str(concept_id))
    return {key: len(ids) for key, ids in counts.items()}


def token_label_share(index: Any) -> dict[str, float]:
    """Token -> the share of this ontology's records whose labels contain it."""
    records = getattr(index, "records", {}) or {}
    total = len(records)
    if not total:
        return {}
    seen: dict[str, int] = {}
    for record in records.values():
        if not isinstance(record, dict):
            continue
        here = {
            token.folded
            for phrase in surface_forms(record)
            for token in tokenize(phrase)
        }
        for token in here:
            seen[token] = seen.get(token, 0) + 1
    return {token: count / total for token, count in seen.items()}


def build_lexicon(
    sources: Iterable[tuple[Any, str]],
    *,
    max_tokens: int = 8,
    max_concepts_per_form: int = MAX_CONCEPTS_PER_FORM,
    max_single_token_label_share: float = MAX_SINGLE_TOKEN_LABEL_SHARE,
    skip_suspect_labels: bool = True,
) -> AliasLexicon:
    """`(index, organizer_type)` pairs -> one lexicon, filtered by governed evidence.

    Very long surface forms are dropped: a governed description of eight or more words is a
    definition, not something a clinician writes in a note, and matching one would only ever
    fire on a coincidence.

    Two further gates come from the knowledge base rather than from a word list: a form that
    names more concepts than `max_concepts_per_form` cannot identify which one a mention
    means, and a record the index marks as having a suspect label is not used at all.
    Finally a one-word form that occurs in more than `max_single_token_label_share` of this
    ontology's labels is a common word of the vocabulary, not a term - also measured here
    rather than listed anywhere.
    """
    pairs = list(sources)
    ambiguity = form_ambiguity(pairs)
    lexicon = AliasLexicon(ambiguity=ambiguity)
    for index, entity_type in pairs:
        share = token_label_share(index)
        records = getattr(index, "records", {}) or {}
        for record in records.values():
            if not isinstance(record, dict):
                continue
            if skip_suspect_labels and label_is_suspect(record):
                lexicon._skip(REJECT_SUSPECT_LABEL)
                continue
            for phrase in surface_forms(record):
                tokens = tokenize(phrase)
                if len(tokens) > max_tokens:
                    lexicon._skip("too_many_tokens")
                    continue
                if ambiguity.get(form_key(phrase), 1) > max_concepts_per_form:
                    lexicon._skip(REJECT_AMBIGUOUS_FORM)
                    continue
                if (
                    len(tokens) == 1
                    and share.get(tokens[0].folded, 0.0) > max_single_token_label_share
                ):
                    lexicon._skip(REJECT_COMMON_TOKEN)
                    continue
                lexicon.add(phrase, entity_type)
    return lexicon


__all__ = [
    "MAX_CONCEPTS_PER_FORM",
    "MAX_SINGLE_TOKEN_LABEL_SHARE",
    "REJECT_COMMON_TOKEN",
    "MIN_ALIAS_CHARS",
    "REJECT_AMBIGUOUS_FORM",
    "REJECT_SUSPECT_LABEL",
    "SHORT_TOKEN_MAX",
    "AliasEntry",
    "AliasHit",
    "AliasLexicon",
    "Token",
    "build_lexicon",
    "find_alias_hits",
    "form_ambiguity",
    "form_key",
    "label_is_suspect",
    "token_label_share",
    "fold",
    "surface_forms",
    "tokenize",
]
