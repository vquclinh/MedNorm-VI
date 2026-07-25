"""Slow-tokenizer (PhoBERT/ViHealthBERT-Word) character alignment for S1.

``demdecuong/vihealthbert-base-word`` declares ``tokenizer_class =
PhobertTokenizer``, which has **no fast (Rust) implementation**: ``use_fast=True``
silently returns the slow tokenizer and ``tokenizer.is_fast`` is ``False``. The
slow tokenizer therefore cannot provide ``return_offsets_mapping``,
``word_ids()``, ``token_to_chars()`` or ``char_to_token()``.

This module reconstructs the same information deterministically, without those
APIs, in three explicit stages:

1. **word segmentation → original characters.** ViHealthBERT-Word expects
   pre-word-segmented Vietnamese input (RDRSegmenter style, syllables joined by
   ``_``). :func:`map_segmented_words` maps each segmented word back to a
   half-open ``[start, end)`` span of the ORIGINAL text using a monotonic cursor
   (never a global ``str.find``), so repeated words, repeated whitespace,
   newlines and punctuation are handled unambiguously.
2. **words → BPE subtokens.** :func:`align_subtokens` tokenizes each segmented
   word with the real slow tokenizer and attaches the word's original character
   span to every piece.
3. **subtokens → supervision.** The caller derives labels from the original
   character spans, exactly as the fast-tokenizer ``offset_mapping`` path did.

All functions are pure and deterministic and never emit raw clinical text.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Protocol

ALIGNMENT_BACKEND = "phobert-slow-char-alignment-v1"

# Subtoken supervision policy (see module docstring / Audit 0022): every
# non-special subtoken inherits the ORIGINAL character span of its segmented
# word, and labels are assigned by character overlap with gold entities — the
# identical rule the fast ``offset_mapping`` path used. Special tokens and
# padding carry a zero label and ``label_mask = 0``.
SUBTOKEN_SUPERVISION_POLICY = "all_subtokens_inherit_word_span_overlap_labels"

# Policy applied when word segmentation merges across a gold entity boundary.
BOUNDARY_MERGE_POLICY = "unalignable_example_counted_and_skipped"

# Policy applied when truncation cuts an entity. An entity whose supervised
# subtokens only PARTIALLY survive must never keep partial labels: every retained
# subtoken of that entity is masked out (label_mask = 0, labels zeroed) and the
# entity is counted. Fully dropped entities are counted separately.
PARTIAL_TRUNCATION_POLICY = "mask_all_retained_subtokens_of_partially_truncated_entity"


class SlowTokenizerLike(Protocol):
    """Minimal slow-tokenizer surface used by the alignment backend."""

    def tokenize(self, text: str) -> list[str]: ...
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...
    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]: ...


class AlignmentError(ValueError):
    """Raised when segmented words cannot be mapped unambiguously."""


@dataclass(frozen=True, slots=True)
class SegmentedWord:
    """One RDRSegmenter word and its span in the ORIGINAL text."""

    model_text: str                 # e.g. "đái_tháo_đường" (fed to the tokenizer)
    original_start: int
    original_end: int               # half-open
    source_word_indices: tuple[int, ...] = field(default_factory=tuple)

    @property
    def length(self) -> int:
        return self.original_end - self.original_start


@dataclass(frozen=True, slots=True)
class AlignedSubtoken:
    """One BPE piece carrying its source word and original character span."""

    token: str
    token_id: int
    original_start: int
    original_end: int               # half-open; (0, 0) for special tokens
    source_word_index: int          # -1 for special tokens
    is_special: bool = False
    is_continuation: bool = False


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    subtokens: tuple[AlignedSubtoken, ...]
    truncated: bool = False
    truncated_entity_count: int = 0


def _strip_accents_fold(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def segmented_text_to_words(segmented_text: str) -> list[str]:
    """Split RDRSegmenter output into words (whitespace separated)."""
    return [w for w in segmented_text.split() if w]


def map_segmented_words(original_text: str, segmented_words: list[str]) -> list[SegmentedWord]:
    """Map each segmented word back to a half-open span of ``original_text``.

    Uses a strictly monotonic cursor: each word's surface form (underscores
    expanded back to their original separators) is located at or after the
    previous word's end, so repeated identical words map to distinct, ordered
    spans. Raises :class:`AlignmentError` when a word cannot be located.
    """
    text = _strip_accents_fold(original_text)
    cursor = 0
    out: list[SegmentedWord] = []
    for index, word in enumerate(segmented_words):
        syllables = [s for s in word.split("_") if s]
        if not syllables:
            raise AlignmentError(f"empty segmented word at index {index}")
        start: int | None = None
        end = cursor
        for position, syllable in enumerate(syllables):
            found = text.find(syllable, end)
            if found < 0:
                raise AlignmentError(
                    f"segmented syllable #{position} of word {index} not found "
                    "in the original text after the current cursor")
            if position > 0:
                gap = text[end:found]
                if gap.strip():
                    # Only whitespace may separate syllables of one segmented word.
                    raise AlignmentError(
                        f"non-whitespace gap inside segmented word {index}")
            if start is None:
                start = found
            end = found + len(syllable)
        assert start is not None
        out.append(SegmentedWord(
            model_text=word, original_start=start, original_end=end,
            source_word_indices=(index,)))
        cursor = end
    return out


def find_boundary_violations(
    words: list[SegmentedWord], entities: list[tuple[int, int]],
) -> list[int]:
    """Indices of words that straddle a gold entity boundary.

    A violation means one model token would have to carry two incompatible
    labels (inside and outside an entity). Callers apply
    :data:`BOUNDARY_MERGE_POLICY`.
    """
    violations: list[int] = []
    for i, word in enumerate(words):
        for ent_start, ent_end in entities:
            overlaps = word.original_start < ent_end and ent_start < word.original_end
            contained = ent_start <= word.original_start and word.original_end <= ent_end
            if overlaps and not contained:
                violations.append(i)
                break
    return violations


def align_subtokens(
    words: list[SegmentedWord], tokenizer: SlowTokenizerLike, *,
    max_length: int, cls_token_id: int | None = None, sep_token_id: int | None = None,
    cls_token: str = "<s>", sep_token: str = "</s>",
) -> AlignmentResult:
    """Tokenize each segmented word with the slow tokenizer and align pieces.

    Every non-special piece inherits its word's ORIGINAL character span. Special
    tokens are added using the tokenizer's real contract when available. The
    sequence is truncated to ``max_length`` (including special tokens); the
    result reports whether truncation occurred.
    """
    if max_length < 2:
        raise AlignmentError("max_length must leave room for special tokens")
    body: list[AlignedSubtoken] = []
    budget = max_length - 2  # reserve CLS/SEP
    truncated = False
    for word_index, word in enumerate(words):
        pieces = tokenizer.tokenize(word.model_text)
        if not pieces:
            continue
        ids = tokenizer.convert_tokens_to_ids(pieces)
        if len(ids) != len(pieces):
            raise AlignmentError("tokenizer returned mismatched pieces/ids")
        for offset, (piece, token_id) in enumerate(zip(pieces, ids, strict=True)):
            if len(body) >= budget:
                truncated = True
                break
            body.append(AlignedSubtoken(
                token=piece, token_id=int(token_id),
                original_start=word.original_start, original_end=word.original_end,
                source_word_index=word_index, is_special=False,
                is_continuation=offset > 0))
        if truncated:
            break
    cls_id = cls_token_id if cls_token_id is not None else 0
    sep_id = sep_token_id if sep_token_id is not None else 2
    sequence = [
        AlignedSubtoken(cls_token, int(cls_id), 0, 0, -1, is_special=True),
        *body,
        AlignedSubtoken(sep_token, int(sep_id), 0, 0, -1, is_special=True),
    ]
    return AlignmentResult(subtokens=tuple(sequence), truncated=truncated)


@dataclass(frozen=True, slots=True)
class TruncationReport:
    """How truncation affected each gold entity.

    ``partially_truncated`` lists indices into the caller's ``entities`` sequence
    whose supervised subtokens only partially survived; those entities must be
    masked out entirely (see :data:`PARTIAL_TRUNCATION_POLICY`).
    """

    fully_retained: tuple[int, ...] = field(default_factory=tuple)
    fully_dropped: tuple[int, ...] = field(default_factory=tuple)
    partially_truncated: tuple[int, ...] = field(default_factory=tuple)

    @property
    def fully_dropped_count(self) -> int:
        return len(self.fully_dropped)

    @property
    def partially_truncated_count(self) -> int:
        return len(self.partially_truncated)

    @property
    def truncated_entity_count(self) -> int:
        """Entities that lost supervision entirely (dropped or masked as partial)."""
        return len(self.fully_dropped) + len(self.partially_truncated)


def classify_truncated_entities(
    words: list[SegmentedWord], result: AlignmentResult, entities: list[tuple[int, int]],
) -> TruncationReport:
    """Classify each entity as fully retained, fully dropped, or partially cut.

    An entity is *fully retained* only when every segmented word overlapping it
    kept at least one subtoken in the (possibly truncated) sequence. If some but
    not all of those words survived, the entity is *partially truncated*.
    """
    surviving_words = {s.source_word_index for s in result.subtokens if not s.is_special}
    retained: list[int] = []
    dropped: list[int] = []
    partial: list[int] = []
    for index, (ent_start, ent_end) in enumerate(entities):
        overlapping = [
            i for i, w in enumerate(words)
            if w.original_start < ent_end and ent_start < w.original_end
        ]
        if not overlapping:
            dropped.append(index)
            continue
        kept = [i for i in overlapping if i in surviving_words]
        if not kept:
            dropped.append(index)
        elif len(kept) == len(overlapping):
            retained.append(index)
        else:
            partial.append(index)
    return TruncationReport(
        fully_retained=tuple(retained), fully_dropped=tuple(dropped),
        partially_truncated=tuple(partial))


def count_truncated_entities(
    result: AlignmentResult, entities: list[tuple[int, int]],
) -> int:
    """Entities with no surviving supervised subtoken after truncation."""
    covered = [(s.original_start, s.original_end) for s in result.subtokens if not s.is_special]
    count = 0
    for ent_start, ent_end in entities:
        if not any(start < ent_end and ent_start < end for start, end in covered):
            count += 1
    return count


def verify_tokenizer_equivalence(
    words: list[SegmentedWord], tokenizer: Any,
) -> dict[str, Any]:
    """Prove per-word alignment IDs equal whole-sentence tokenization.

    The alignment backend tokenizes **word by word** to attach character spans.
    This helper re-tokenizes the FULL segmented sentence with the real tokenizer
    (``add_special_tokens=False``) and asserts the concatenated per-word IDs match
    exactly — catching any tokenizer that is not per-word decomposable.

    Returns an aggregate, text-free report; raises :class:`AlignmentError` on
    mismatch so callers can fail before acquiring model weights.
    """
    per_word_ids: list[int] = []
    for word in words:
        pieces = tokenizer.tokenize(word.model_text)
        per_word_ids.extend(int(v) for v in tokenizer.convert_tokens_to_ids(pieces))
    sentence = " ".join(word.model_text for word in words)
    try:
        encoded = tokenizer(sentence, add_special_tokens=False)
        sentence_ids = [int(v) for v in encoded["input_ids"]]
    except TypeError:
        pieces = tokenizer.tokenize(sentence)
        sentence_ids = [int(v) for v in tokenizer.convert_tokens_to_ids(pieces)]
    if per_word_ids != sentence_ids:
        raise AlignmentError(
            "tokenizer equivalence failed: per-word alignment produced "
            f"{len(per_word_ids)} ids but whole-sentence tokenization produced "
            f"{len(sentence_ids)}; the tokenizer is not per-word decomposable")
    return {
        "equivalent": True,
        "word_count": len(words),
        "token_count": len(per_word_ids),
    }


def describe_backend(tokenizer: Any) -> dict[str, Any]:
    """Aggregate, text-free description of the tokenizer/alignment backend."""
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "alignment_backend": ALIGNMENT_BACKEND,
        "subtoken_supervision_policy": SUBTOKEN_SUPERVISION_POLICY,
        "boundary_merge_policy": BOUNDARY_MERGE_POLICY,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
    }


__all__ = [
    "ALIGNMENT_BACKEND",
    "SUBTOKEN_SUPERVISION_POLICY",
    "BOUNDARY_MERGE_POLICY",
    "PARTIAL_TRUNCATION_POLICY",
    "AlignmentError",
    "AlignedSubtoken",
    "AlignmentResult",
    "SegmentedWord",
    "SlowTokenizerLike",
    "TruncationReport",
    "align_subtokens",
    "classify_truncated_entities",
    "count_truncated_entities",
    "describe_backend",
    "find_boundary_violations",
    "map_segmented_words",
    "segmented_text_to_words",
    "verify_tokenizer_equivalence",
]
