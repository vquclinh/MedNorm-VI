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

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

ALIGNMENT_BACKEND = "phobert-slow-char-alignment-v4"

# The character RDRSegmenter uses to join the syllables of one word. Some governed
# sources (ViMQ, PhoNER-COVID19) are distributed ALREADY word-segmented, so this
# character also occurs literally in ``original_text``. The two cases are not
# distinguishable from the segmenter output alone, so both are accepted when
# locating a word's syllables (Audit 0026).
SEGMENTER_JOIN_CHARACTER = "_"

# Reason codes for an example that could not be encoded. UNEXPECTED codes are
# implementation/segmentation failures and MUST block full-training readiness.
# EXPECTED codes are deterministic, tracked governance decisions and must not.
REASON_EMPTY_SEGMENTED_WORD = "EMPTY_SEGMENTED_WORD"
REASON_SEGMENTED_SYLLABLE_NOT_FOUND = "SEGMENTED_SYLLABLE_NOT_FOUND"
REASON_NON_SEPARATOR_GAP_INSIDE_WORD = "NON_SEPARATOR_GAP_INSIDE_WORD"
REASON_TOKENIZER_PIECE_ID_MISMATCH = "TOKENIZER_PIECE_ID_MISMATCH"
REASON_TOKENIZER_NOT_DECOMPOSABLE = "TOKENIZER_NOT_DECOMPOSABLE"
REASON_ENTITY_OFFSET_INVARIANT_VIOLATED = "ENTITY_OFFSET_INVARIANT_VIOLATED"
REASON_MAX_LENGTH_TOO_SMALL = "MAX_LENGTH_TOO_SMALL"
REASON_SEGMENTER_ALTERED_CHARACTERS = "SEGMENTER_ALTERED_CHARACTERS"
REASON_GOVERNED_EXCLUSION = "GOVERNED_EXCLUSION"

UNEXPECTED_REASON_CODES = (
    REASON_EMPTY_SEGMENTED_WORD,
    REASON_SEGMENTED_SYLLABLE_NOT_FOUND,
    REASON_NON_SEPARATOR_GAP_INSIDE_WORD,
    REASON_TOKENIZER_PIECE_ID_MISMATCH,
    REASON_TOKENIZER_NOT_DECOMPOSABLE,
    REASON_ENTITY_OFFSET_INVARIANT_VIOLATED,
    REASON_MAX_LENGTH_TOO_SMALL,
    REASON_SEGMENTER_ALTERED_CHARACTERS,
)
EXPECTED_REASON_CODES = (REASON_GOVERNED_EXCLUSION,)

# Pipeline stage at which an example failed; kept coarse so it never leaks text.
STAGE_WORD_MAPPING = "word_mapping"
STAGE_TOKENIZER_EQUIVALENCE = "tokenizer_equivalence"
STAGE_SUBTOKEN_ENCODING = "subtoken_encoding"
STAGE_GOVERNED_EXCLUSION = "governed_exclusion"

# Subtoken supervision policy (see module docstring / Audit 0022): every
# non-special subtoken inherits the ORIGINAL character span of its segmented
# word, and labels are assigned by character overlap with gold entities — the
# identical rule the fast ``offset_mapping`` path used. Special tokens and
# padding carry a zero label and ``label_mask = 0``.
SUBTOKEN_SUPERVISION_POLICY = "all_subtokens_inherit_word_span_overlap_labels"

# Policy applied when word segmentation merges across a gold entity boundary.
# Audit 0026: previously the whole example was discarded, which threw away every
# other entity in it over a single ambiguous word. The straddling word and every
# subtoken of the entities it touches are now masked instead (labels zeroed,
# label_mask 0) and counted — the same rule PARTIAL_TRUNCATION_POLICY already
# applies to truncation. No token is ever given an incompatible label.
BOUNDARY_MERGE_POLICY = "mask_straddling_word_and_affected_entity_subtokens"

# Policy applied when truncation cuts an entity. An entity whose supervised
# subtokens only PARTIALLY survive must never keep partial labels: every retained
# subtoken of that entity is masked out (label_mask = 0, labels zeroed) and the
# entity is counted. Fully dropped entities are counted separately.
PARTIAL_TRUNCATION_POLICY = "mask_all_retained_subtokens_of_partially_truncated_entity"


# How the segmented form of one example was obtained.
SEGMENTATION_SOURCE_PRE_SEGMENTED = "pre_segmented_source"
SEGMENTATION_SOURCE_SEGMENTER = "vncorenlp_rdrsegmenter"

# A join character sitting directly between two word characters is the signature
# of RDRSegmenter output. Some governed sources (ViMQ, PhoNER-COVID19) ship text
# that is ALREADY segmented this way; ``[^\W_]`` is "word character except the
# join character itself".
PRE_SEGMENTED_PATTERN = re.compile(r"[^\W_]_[^\W_]", re.UNICODE)


def looks_pre_segmented(text: str) -> bool:
    """True when ``text`` is already RDRSegmenter output.

    Re-segmenting already-segmented text is destructive: RDRSegmenter splits the
    join character off as a standalone token, so a single word like a two-syllable
    compound is shredded into three model tokens and the word-level input
    ViHealthBERT-Word expects is lost.
    """
    return bool(PRE_SEGMENTED_PATTERN.search(str(text)))


def resolve_segmented_text(
    original_text: str, segmenter: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    """Return ``(segmented_text, segmentation_source)`` for one example.

    Already-segmented text is used verbatim; everything else goes through the
    supplied production segmenter. This is the single policy both the smoke and
    the full-training path use.
    """
    if looks_pre_segmented(original_text):
        return str(original_text), SEGMENTATION_SOURCE_PRE_SEGMENTED
    if segmenter is None:
        return str(original_text), SEGMENTATION_SOURCE_PRE_SEGMENTED
    return str(segmenter(original_text)), SEGMENTATION_SOURCE_SEGMENTER


class SlowTokenizerLike(Protocol):
    """Minimal slow-tokenizer surface used by the alignment backend."""

    def tokenize(self, text: str) -> list[str]: ...
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...
    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]: ...


class AlignmentError(ValueError):
    """Raised when segmented words cannot be mapped unambiguously.

    ``reason_code`` is a stable, text-free classifier (see the REASON_* constants)
    so callers can record *why* an example failed without ever touching its
    content, and can separate unexpected failures from governed exclusions.
    """

    def __init__(self, message: str, reason_code: str = "") -> None:
        super().__init__(message)
        self.reason_code = reason_code or REASON_SEGMENTED_SYLLABLE_NOT_FOUND


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


# Vietnamese tone-mark placement equivalence (Audit 0028).
#
# In an oa / oe / uy cluster the tone mark may sit on either vowel: "tone on the
# first vowel" and "tone on the second vowel" are two spellings of the SAME word,
# and the mapping between them is a bijection over exactly two code points per
# side. RDRSegmenter normalizes to the traditional placement (tone on the second
# vowel): measured over its own output in the governed corpus, 4,700 such clusters
# are traditional and ZERO are the alternative form, while raw source text runs
# roughly 23% alternative.
#
# Accepting this equivalence is offset-safe because both sides are exactly two
# code points, so the cursor advances equally and every span still indexes
# untouched ``original_text``. It is NOT fuzzy matching: only these listed pairs
# are equivalent, and every other difference still fails.
TONE_PLACEMENT_EQUIVALENTS: tuple[tuple[str, str], ...] = (
    ("oà", "òa"), ("oá", "óa"), ("oả", "ỏa"), ("oã", "õa"), ("oạ", "ọa"),
    ("oè", "òe"), ("oé", "óe"), ("oẻ", "ỏe"), ("oẽ", "õe"), ("oẹ", "ọe"),
    ("uỳ", "ùy"), ("uý", "úy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy"),
)


def _equivalent_cluster_table() -> frozenset[tuple[str, str]]:
    """Both directions and both letter cases of every documented pair."""
    table: set[tuple[str, str]] = set()
    for first, second in TONE_PLACEMENT_EQUIVALENTS:
        for left, right in ((first, second), (second, first)):
            table.add((left, right))
            table.add((left.capitalize(), right.capitalize()))
            table.add((left.upper(), right.upper()))
    return frozenset(table)


EQUIVALENT_CLUSTERS = _equivalent_cluster_table()
EQUIVALENT_CLUSTER_LENGTH = 2


def equivalent_cluster_length(
    original: str, original_index: int, segmented: str, segmented_index: int,
) -> int:
    """Length of a documented orthographic-equivalent cluster, else ``0``.

    Returns the SAME length for both sides, which is what keeps the alignment
    offset-exact.
    """
    pair = (
        original[original_index:original_index + EQUIVALENT_CLUSTER_LENGTH],
        segmented[segmented_index:segmented_index + EQUIVALENT_CLUSTER_LENGTH],
    )
    return EQUIVALENT_CLUSTER_LENGTH if pair in EQUIVALENT_CLUSTERS else 0


def _mismatch_detail(original_char: str, segmented_char: str) -> str:
    """Structural description of a divergence. Never reports the characters.

    ``same_base_letter`` separates a diacritic-level rewrite from a genuinely
    different letter, which is the distinction that matters when triaging.
    """
    original_base = unicodedata.normalize("NFD", original_char)[:1]
    segmented_base = unicodedata.normalize("NFD", segmented_char)[:1]
    return (
        f"categories {unicodedata.category(original_char)}/"
        f"{unicodedata.category(segmented_char)}, "
        f"same_base_letter={original_base == segmented_base}"
    )


def is_separator_character(char: str) -> bool:
    """Characters a segmenter may freely insert, remove, or move.

    Whitespace and :data:`SEGMENTER_JOIN_CHARACTER` carry no supervision: the
    segmenter inserts the join character between syllables it groups, and it may
    add or drop whitespace around punctuation. Every OTHER character must survive
    segmentation unchanged, which is what makes offset reconstruction possible.
    """
    return char.isspace() or char == SEGMENTER_JOIN_CHARACTER


def is_legal_syllable_gap(gap: str) -> bool:
    """May ``gap`` separate two syllables of one segmented word?"""
    return all(is_separator_character(char) for char in gap)


def map_segmented_words(original_text: str, segmented_words: list[str]) -> list[SegmentedWord]:
    """Map each segmented word back to a half-open span of ``original_text``.

    A word segmenter is allowed to regroup text: it inserts the join character
    between syllables it merges, and it may add or drop whitespace (for example
    pulling a hyphenated compound together, or splitting a standalone join
    character off as its own token). What it must NOT do is change, add, or drop a
    real character.

    So the mapping walks both strings with a strictly monotonic cursor and matches
    them **character by character**, skipping separators on either side. Each
    word's span runs from its first to its last matched original character. This
    reconstructs exact offsets without assuming any particular regrouping, and it
    still fails loudly the moment a real character does not survive.

    Raises :class:`AlignmentError` when the character sequences diverge.
    """
    text = _strip_accents_fold(original_text)
    cursor = 0
    out: list[SegmentedWord] = []
    for index, word in enumerate(segmented_words):
        start: int | None = None
        end = 0
        position = 0
        while position < len(word):
            char = word[position]
            if is_separator_character(char):
                position += 1                 # inserted by the segmenter, or a
                continue                      # literal separator it may have moved
            while cursor < len(text) and is_separator_character(text[cursor]):
                cursor += 1
            if cursor >= len(text):
                raise AlignmentError(
                    f"segmentation ran past the end of the original text at word "
                    f"{index}, character {position}",
                    REASON_SEGMENTED_SYLLABLE_NOT_FOUND)
            if text[cursor] == char:
                if start is None:
                    start = cursor
                end = cursor + 1
                cursor += 1
                position += 1
                continue
            # A documented orthographic equivalence (same word, tone mark on the
            # other vowel) consumes the SAME number of code points on both sides,
            # so offsets stay exact.
            span = equivalent_cluster_length(text, cursor, word, position)
            if span:
                if start is None:
                    start = cursor
                end = cursor + span
                cursor += span
                position += span
                continue
            # Structural, text-free detail: WHERE it diverged and WHAT KIND of
            # character each side had. The characters themselves are never
            # reported, so no clinical content can leak into a diagnostic.
            raise AlignmentError(
                f"segmenter altered a character: word {index} character {position} "
                f"does not match original offset {cursor} "
                f"({_mismatch_detail(text[cursor], char)})",
                REASON_SEGMENTER_ALTERED_CHARACTERS)
        if start is None:
            # A token made only of separators - RDRSegmenter emits a standalone
            # join character when it re-segments already-segmented text. It carries
            # no characters of its own, so it contributes no word and no label.
            continue
        out.append(SegmentedWord(
            model_text=word, original_start=start, original_end=end,
            source_word_indices=(index,)))
    if not out:
        raise AlignmentError(
            "segmentation produced no mappable words", REASON_EMPTY_SEGMENTED_WORD)
    # Anything left over in the original that is not a separator was dropped by the
    # segmenter. Ignoring it would silently strip supervision from the tail of an
    # example, so it is a failure like any other altered character.
    remaining = [
        offset for offset in range(cursor, len(text))
        if not is_separator_character(text[offset])
    ]
    if remaining:
        raise AlignmentError(
            f"segmenter dropped {len(remaining)} original character(s) starting at "
            f"offset {remaining[0]} (unicode category "
            f"{unicodedata.category(text[remaining[0]])})",
            REASON_SEGMENTER_ALTERED_CHARACTERS)
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


def entities_touched_by_boundary_merge(
    words: list[SegmentedWord], entities: list[tuple[int, int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Straddling word indices and the entity indices they make ambiguous.

    Applies :data:`BOUNDARY_MERGE_POLICY`: a word that overlaps an entity without
    being contained in it would need two incompatible labels, so that word and
    every entity it touches lose supervision instead of being guessed at.
    """
    straddling: list[int] = []
    affected: set[int] = set()
    for word_index, word in enumerate(words):
        straddles = False
        for entity_index, (ent_start, ent_end) in enumerate(entities):
            overlaps = word.original_start < ent_end and ent_start < word.original_end
            contained = ent_start <= word.original_start and word.original_end <= ent_end
            if overlaps and not contained:
                straddles = True
                affected.add(entity_index)
        if straddles:
            straddling.append(word_index)
            # Any entity this word also fully covers becomes ambiguous too.
            for entity_index, (ent_start, ent_end) in enumerate(entities):
                if word.original_start < ent_end and ent_start < word.original_end:
                    affected.add(entity_index)
    return tuple(straddling), tuple(sorted(affected))


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
        raise AlignmentError(
            "max_length must leave room for special tokens", REASON_MAX_LENGTH_TOO_SMALL)
    body: list[AlignedSubtoken] = []
    budget = max_length - 2  # reserve CLS/SEP
    truncated = False
    for word_index, word in enumerate(words):
        pieces = tokenizer.tokenize(word.model_text)
        if not pieces:
            continue
        ids = tokenizer.convert_tokens_to_ids(pieces)
        if len(ids) != len(pieces):
            raise AlignmentError(
                "tokenizer returned mismatched pieces/ids", REASON_TOKENIZER_PIECE_ID_MISMATCH)
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
            f"{len(sentence_ids)}; the tokenizer is not per-word decomposable",
            REASON_TOKENIZER_NOT_DECOMPOSABLE)
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
    "SUBTOKEN_SUPERVISION_POLICY",
    "BOUNDARY_MERGE_POLICY",
    "PARTIAL_TRUNCATION_POLICY",
    "ALIGNMENT_BACKEND",
    "EXPECTED_REASON_CODES",
    "SEGMENTATION_SOURCE_PRE_SEGMENTED",
    "SEGMENTATION_SOURCE_SEGMENTER",
    "SEGMENTER_JOIN_CHARACTER",
    "TONE_PLACEMENT_EQUIVALENTS",
    "EQUIVALENT_CLUSTERS",
    "STAGE_GOVERNED_EXCLUSION",
    "STAGE_SUBTOKEN_ENCODING",
    "STAGE_TOKENIZER_EQUIVALENCE",
    "STAGE_WORD_MAPPING",
    "UNEXPECTED_REASON_CODES",
    "REASON_EMPTY_SEGMENTED_WORD",
    "REASON_ENTITY_OFFSET_INVARIANT_VIOLATED",
    "REASON_GOVERNED_EXCLUSION",
    "REASON_MAX_LENGTH_TOO_SMALL",
    "REASON_NON_SEPARATOR_GAP_INSIDE_WORD",
    "REASON_SEGMENTED_SYLLABLE_NOT_FOUND",
    "REASON_SEGMENTER_ALTERED_CHARACTERS",
    "REASON_TOKENIZER_NOT_DECOMPOSABLE",
    "REASON_TOKENIZER_PIECE_ID_MISMATCH",
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
    "entities_touched_by_boundary_merge",
    "equivalent_cluster_length",
    "find_boundary_violations",
    "is_legal_syllable_gap",
    "is_separator_character",
    "looks_pre_segmented",
    "resolve_segmented_text",
    "map_segmented_words",
    "segmented_text_to_words",
    "verify_tokenizer_equivalence",
]
