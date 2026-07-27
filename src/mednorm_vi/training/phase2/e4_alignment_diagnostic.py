"""Full-corpus E4 word-alignment diagnostic (Audit 0038).

Scans the complete governed train and validation splits and reports, for every
gold entity, whether it aligns to

* the **VnCoreNLP segmented model-word** surface — the Audit-0037 grid, and
* the **atomic original-text word** surface — the Audit-0038 grid.

The point of the scan is to replace an assumption with a count: how many governed
entities the segmented surface cannot represent, how many the atomic surface
fixes, and how many remain unalignable. An entity that still does not align is
**reported, never repaired** — no snapping, no trimming, no silent exclusion.

Nothing here trains, and internal_test is never opened.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...mention_factory.w2ner import (
    ATOMIC_WORD_POLICY_VERSION,
    EntitySpan,
    WordToken,
    entity_atomic_alignment,
    tokenize_atomic_words,
)
from ..phobert_alignment import (
    SegmentedWord,
    map_segmented_words,
    resolve_segmented_text,
    segmented_text_to_words,
)

DIAGNOSTIC_VERSION = "e4-alignment-diagnostic-v1"

# internal_test is frozen. The diagnostic refuses to open it at all.
FORBIDDEN_SPLITS = frozenset({"internal_test"})

ALIGNED = "aligned"
LEFT_ONLY = "misaligned_left_only"
RIGHT_ONLY = "misaligned_right_only"
BOTH = "misaligned_both"

ALIGNMENT_CATEGORIES: tuple[str, ...] = (ALIGNED, LEFT_ONLY, RIGHT_ONLY, BOTH)


class E4DiagnosticError(RuntimeError):
    """Raised when the diagnostic is asked to do something it must not."""


def classify_alignment(words: Sequence[WordToken], entity: EntitySpan) -> str:
    left, right = entity_atomic_alignment(words, entity)
    if left and right:
        return ALIGNED
    if left:
        return RIGHT_ONLY
    if right:
        return LEFT_ONLY
    return BOTH


@dataclass(frozen=True, slots=True)
class MismatchRecord:
    """One entity the segmented model-word surface could not represent."""

    example_id: str
    split: str
    row_index: int
    source: str
    entity_type: str
    start: int
    end: int
    segmented_category: str
    atomic_category: str
    straddling_model_word: str = ""
    straddling_model_word_start: int = -1
    straddling_model_word_end: int = -1

    @property
    def fixed_by_atomic_words(self) -> bool:
        return self.segmented_category != ALIGNED and self.atomic_category == ALIGNED

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "row_index": self.row_index,
            "source": self.source,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "segmented_category": self.segmented_category,
            "atomic_category": self.atomic_category,
            "fixed_by_atomic_words": self.fixed_by_atomic_words,
            "straddling_model_word": self.straddling_model_word,
            "straddling_model_word_start": self.straddling_model_word_start,
            "straddling_model_word_end": self.straddling_model_word_end,
        }


@dataclass
class _Counters:
    examples: int = 0
    entities: int = 0
    segmented: dict[str, int] = field(default_factory=lambda: dict.fromkeys(
        ALIGNMENT_CATEGORIES, 0))
    atomic: dict[str, int] = field(default_factory=lambda: dict.fromkeys(
        ALIGNMENT_CATEGORIES, 0))

    def bump(self, segmented_category: str, atomic_category: str) -> None:
        self.entities += 1
        self.segmented[segmented_category] = self.segmented.get(segmented_category, 0) + 1
        self.atomic[atomic_category] = self.atomic.get(atomic_category, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "entities": self.entities,
            "segmented_model_word_surface": dict(self.segmented),
            "atomic_original_word_surface": dict(self.atomic),
        }


@dataclass(frozen=True, slots=True)
class E4AlignmentDiagnostic:
    """The deterministic, fully-reported outcome of the corpus scan."""

    payload: dict[str, Any]

    @property
    def unalignable_after_atomic(self) -> int:
        return int(self.payload["totals"]["entities_unalignable_after_atomic_words"])

    @property
    def silent_exclusions(self) -> int:
        return int(self.payload["totals"]["silent_exclusions"])

    @property
    def projection_violations(self) -> int:
        return int(self.payload["totals"]["atomic_projection_violations"])

    @property
    def passed(self) -> bool:
        """Governed policy: zero unalignable, zero exclusions, zero projection breaks."""
        return (
            self.unalignable_after_atomic == 0
            and self.silent_exclusions == 0
            and self.projection_violations == 0
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def write(self, path: str | Path) -> str:
        import hashlib

        raw = json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(raw, encoding="utf-8")
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, Any]:
        """Compact console/report summary."""
        totals = self.payload["totals"]
        return {
            "diagnostic_version": self.payload["diagnostic_version"],
            "examples": totals["examples"],
            "entities": totals["entities"],
            "segmented_aligned": totals["segmented_aligned"],
            "segmented_misaligned_left_only": totals["segmented_misaligned_left_only"],
            "segmented_misaligned_right_only": totals["segmented_misaligned_right_only"],
            "segmented_misaligned_both": totals["segmented_misaligned_both"],
            "entities_fixed_by_atomic_words": totals["entities_fixed_by_atomic_words"],
            "entities_unalignable_after_atomic_words":
                totals["entities_unalignable_after_atomic_words"],
            "silent_exclusions": totals["silent_exclusions"],
            "atomic_projection_violations": totals["atomic_projection_violations"],
            "max_atomic_word_count": totals["max_atomic_word_count"],
            "max_model_word_count": totals["max_model_word_count"],
            "max_phobert_subtoken_count": totals["max_phobert_subtoken_count"],
            "examples_exceeding_max_words": totals["examples_exceeding_max_words"],
            "examples_exceeding_max_model_tokens": totals["examples_exceeding_max_model_tokens"],
            "passed": self.passed,
        }


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                yield index, json.loads(line)


def _entity_from_row(entity: Mapping[str, Any]) -> EntitySpan:
    entity_type = entity.get("target_type") or entity.get("type") or entity.get("label")
    if entity_type is None:
        raise E4DiagnosticError("governed entity has no organizer type field")
    return EntitySpan(
        int(entity["start"]), int(entity["end"]), str(entity_type), str(entity["text"]))


def _straddling(words: Sequence[SegmentedWord], entity: EntitySpan) -> SegmentedWord | None:
    for word in words:
        if (word.original_start < entity.start < word.original_end
                or word.original_start < entity.end < word.original_end):
            return word
    return None


def run_alignment_diagnostic(
    splits: Mapping[str, str | Path],
    *,
    segmenter: Callable[[str], str] | None,
    tokenizer: Any = None,
    max_words: int = 256,
    max_model_tokens: int = 512,
    mismatch_limit: int = 200,
) -> E4AlignmentDiagnostic:
    """Scan governed splits and report every alignment fact, hiding nothing.

    ``tokenizer`` is optional: when it is absent the PhoBERT subtoken statistics are
    reported as unavailable rather than guessed.
    """
    for split in splits:
        if split in FORBIDDEN_SPLITS:
            raise E4DiagnosticError(
                f"the E4 alignment diagnostic must never read {split!r}")

    by_split: dict[str, _Counters] = {}
    by_source: dict[str, _Counters] = {}
    by_type: dict[str, _Counters] = {}
    mismatches: list[MismatchRecord] = []

    examples = 0
    entities = 0
    segmented_counts = dict.fromkeys(ALIGNMENT_CATEGORIES, 0)
    atomic_counts = dict.fromkeys(ALIGNMENT_CATEGORIES, 0)
    fixed_by_atomic = 0
    max_atomic = 0
    max_model = 0
    max_subtokens = -1
    over_max_words = 0
    over_max_tokens = 0
    skipped_examples: list[str] = []
    projection_violations: list[dict[str, Any]] = []

    for split, path in sorted(splits.items()):
        split_counter = by_split.setdefault(split, _Counters())
        for row_index, row in _iter_jsonl(Path(path)):
            examples += 1
            split_counter.examples += 1
            example_id = str(row.get("example_id", row.get("id", row_index)))
            source = str(row.get("source_dataset", ""))
            source_counter = by_source.setdefault(source or "unknown", _Counters())
            source_counter.examples += 1
            text = str(row["text"])

            atomic_words = tokenize_atomic_words(text)
            segmented_text, _origin = resolve_segmented_text(text, segmenter)
            model_words = map_segmented_words(text, segmented_text_to_words(segmented_text))

            max_atomic = max(max_atomic, len(atomic_words))
            max_model = max(max_model, len(model_words))
            if len(atomic_words) > max_words:
                over_max_words += 1
            if tokenizer is not None:
                subtokens = sum(
                    len(tokenizer.tokenize(word.model_text)) for word in model_words)
                max_subtokens = max(max_subtokens, subtokens)
                if subtokens + 2 > max_model_tokens:
                    over_max_tokens += 1

            model_word_tokens = tuple(
                WordToken(index, text[word.original_start:word.original_end],
                          word.original_start, word.original_end)
                for index, word in enumerate(model_words)
            )

            # Projection invariant: every atomic word must OVERLAP at least one
            # model word, otherwise it has no subtokens to pool. Containment is
            # deliberately not required — neither surface refines the other (see
            # AtomicWordProjection).
            for atomic_word in atomic_words:
                if not any(
                    model.original_start < atomic_word.end
                    and atomic_word.start < model.original_end
                    for model in model_words
                ):
                    projection_violations.append({
                        "example_id": example_id,
                        "split": split,
                        "row_index": row_index,
                        "atomic_word_index": atomic_word.index,
                        "start": atomic_word.start,
                        "end": atomic_word.end,
                    })

            for entity_payload in row.get("entities") or []:
                entity = _entity_from_row(entity_payload)
                entity.validate_against(text)
                entities += 1
                segmented_category = classify_alignment(model_word_tokens, entity)
                atomic_category = classify_alignment(atomic_words, entity)
                segmented_counts[segmented_category] += 1
                atomic_counts[atomic_category] += 1
                split_counter.bump(segmented_category, atomic_category)
                source_counter.bump(segmented_category, atomic_category)
                by_type.setdefault(entity.entity_type, _Counters()).bump(
                    segmented_category, atomic_category)
                if segmented_category == ALIGNED:
                    continue
                if atomic_category == ALIGNED:
                    fixed_by_atomic += 1
                straddler = _straddling(model_words, entity)
                if len(mismatches) < mismatch_limit:
                    mismatches.append(MismatchRecord(
                        example_id=example_id, split=split, row_index=row_index,
                        source=source, entity_type=entity.entity_type,
                        start=entity.start, end=entity.end,
                        segmented_category=segmented_category,
                        atomic_category=atomic_category,
                        straddling_model_word=(
                            straddler.model_text if straddler is not None else ""),
                        straddling_model_word_start=(
                            straddler.original_start if straddler is not None else -1),
                        straddling_model_word_end=(
                            straddler.original_end if straddler is not None else -1)))

    unalignable = sum(
        count for category, count in atomic_counts.items() if category != ALIGNED)
    payload: dict[str, Any] = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "atomic_word_policy": ATOMIC_WORD_POLICY_VERSION,
        "splits": sorted(splits),
        "internal_test_accessed": False,
        "max_words": max_words,
        "max_model_tokens": max_model_tokens,
        "totals": {
            "examples": examples,
            "entities": entities,
            "segmented_aligned": segmented_counts[ALIGNED],
            "segmented_misaligned_left_only": segmented_counts[LEFT_ONLY],
            "segmented_misaligned_right_only": segmented_counts[RIGHT_ONLY],
            "segmented_misaligned_both": segmented_counts[BOTH],
            "atomic_aligned": atomic_counts[ALIGNED],
            "atomic_misaligned_left_only": atomic_counts[LEFT_ONLY],
            "atomic_misaligned_right_only": atomic_counts[RIGHT_ONLY],
            "atomic_misaligned_both": atomic_counts[BOTH],
            "entities_fixed_by_atomic_words": fixed_by_atomic,
            "entities_unalignable_after_atomic_words": unalignable,
            "silent_exclusions": len(skipped_examples),
            "max_atomic_word_count": max_atomic,
            "max_model_word_count": max_model,
            "max_phobert_subtoken_count": max_subtokens,
            "phobert_subtoken_statistics_available": tokenizer is not None,
            "examples_exceeding_max_words": over_max_words,
            # Only meaningful when a tokenizer was supplied; -1 means "not measured
            # here", never "zero".
            "examples_exceeding_max_model_tokens": (
                over_max_tokens if tokenizer is not None else -1),
            "atomic_projection_violations": len(projection_violations),
        },
        "by_split": {name: counter.as_dict() for name, counter in sorted(by_split.items())},
        "by_source": {name: counter.as_dict() for name, counter in sorted(by_source.items())},
        "by_entity_type": {name: counter.as_dict() for name, counter in sorted(by_type.items())},
        "mismatch_examples": [record.as_dict() for record in mismatches],
        "mismatch_examples_truncated": len(mismatches) >= mismatch_limit,
        "excluded_examples": list(skipped_examples),
        "atomic_projection_violation_examples": projection_violations[:50],
    }
    return E4AlignmentDiagnostic(payload=payload)


__all__ = [
    "ALIGNED",
    "ALIGNMENT_CATEGORIES",
    "BOTH",
    "DIAGNOSTIC_VERSION",
    "E4AlignmentDiagnostic",
    "E4DiagnosticError",
    "FORBIDDEN_SPLITS",
    "LEFT_ONLY",
    "MismatchRecord",
    "RIGHT_ONLY",
    "classify_alignment",
    "run_alignment_diagnostic",
]
