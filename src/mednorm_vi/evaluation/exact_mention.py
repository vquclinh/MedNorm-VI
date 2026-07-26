"""Exact character-offset mention evaluator for S1 / L4 (Audit 0033).

The S1 training loop reports a token-index span proxy. That is fine for
checkpoint selection but it is not what the organizer scores, and it cannot tell
a boundary slip from a complete miss. This module evaluates mentions the way the
task defines them: on exact ``(start, end, type)`` triples over
``original_text``, with the spec §4 invariant enforced::

    original_text[start:end] == entity["text"]

**Scope.** Mention span and type only. It does **not** reproduce the complete
organizer score: assertions (Jaccard, 30%) and ICD/RxNorm candidate sets
(Jaccard, 40%) are not integrated here, so the numbers below are a mention-level
component, never a leaderboard estimate.

Matching is exact and deterministic. Nothing is fuzzy-matched: a prediction whose
text merely resembles a gold mention is a miss plus a spurious prediction, and is
additionally *categorised* as a boundary error so the failure is explainable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..schemas.constants import ENTITY_TYPES

# Error taxonomy. EXACT_MATCH is the only outcome that scores as a true positive;
# every other category is simultaneously a false positive and/or false negative.
EXACT_MATCH = "exact_match"
MISSED = "missed"
SPURIOUS = "spurious"
WRONG_TYPE = "wrong_type"
LEFT_BOUNDARY = "left_boundary"
RIGHT_BOUNDARY = "right_boundary"
BOTH_BOUNDARY = "both_boundary"
DUPLICATE_OVERLAP = "duplicate_overlap"

ERROR_CATEGORIES = (
    EXACT_MATCH, MISSED, SPURIOUS, WRONG_TYPE,
    LEFT_BOUNDARY, RIGHT_BOUNDARY, BOTH_BOUNDARY, DUPLICATE_OVERLAP,
)

# Provenance buckets, so deterministic and neural contributions stay separable.
PROVENANCE_DETERMINISTIC = "deterministic"
PROVENANCE_NEURAL = "neural"
PROVENANCE_HYBRID = "hybrid"
PROVENANCE_UNKNOWN = "unknown"

# Entity-length buckets in code points, chosen so short symptom mentions and long
# medication phrases land in different cells.
LENGTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1-5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20),
    ("21-40", 21, 40), ("41+", 41, 10**9),
)


class ExactMentionError(ValueError):
    """Raised when an input violates the offset/text invariant."""


@dataclass(frozen=True, slots=True)
class Mention:
    """One mention with exact absolute coordinates into ``original_text``."""

    start: int
    end: int
    entity_type: str
    text: str
    route: str = ""
    section: str = ""
    provenance: str = PROVENANCE_UNKNOWN

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.start, self.end, self.entity_type)

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Mention) -> bool:
        return self.start < other.end and other.start < self.end


def length_bucket(length: int) -> str:
    for name, low, high in LENGTH_BUCKETS:
        if low <= length <= high:
            return name
    return LENGTH_BUCKETS[-1][0]


def privacy_safe_example_id(example_id: str) -> str:
    """Stable, non-reversible handle — never a verbatim id in a tracked file."""
    return hashlib.sha256(str(example_id).encode("utf-8")).hexdigest()[:16]


def validate_mention(mention: Mention, original_text: str, *, role: str) -> None:
    """Enforce spec §4: the span must slice exactly the mention text."""
    if mention.entity_type not in ENTITY_TYPES:
        raise ExactMentionError(
            f"{role} mention has unsupported type {mention.entity_type!r}")
    if mention.start < 0 or mention.end < mention.start:
        raise ExactMentionError(
            f"{role} mention has an invalid span [{mention.start}, {mention.end})")
    if mention.end > len(original_text):
        raise ExactMentionError(
            f"{role} mention end {mention.end} exceeds the text length "
            f"{len(original_text)}")
    if original_text[mention.start:mention.end] != mention.text:
        raise ExactMentionError(
            f"{role} mention violates original_text[start:end] == text at "
            f"[{mention.start}, {mention.end})")


@dataclass
class _Counts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def as_dict(self) -> dict[str, float]:
        precision = (self.true_positive / (self.true_positive + self.false_positive)
                     if self.true_positive + self.false_positive else 0.0)
        recall = (self.true_positive / (self.true_positive + self.false_negative)
                  if self.true_positive + self.false_negative else 0.0)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": precision, "recall": recall, "f1": f1,
        }


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """One privacy-safe diagnostic. Never carries clinical text."""

    privacy_safe_example_id: str
    category: str
    gold_type: str = ""
    predicted_type: str = ""
    gold_span: tuple[int, int] | None = None
    predicted_span: tuple[int, int] | None = None
    route: str = ""
    section: str = ""
    source: str = ""
    provenance: str = PROVENANCE_UNKNOWN

    def as_dict(self) -> dict[str, Any]:
        return {
            "privacy_safe_example_id": self.privacy_safe_example_id,
            "category": self.category,
            "gold_type": self.gold_type,
            "predicted_type": self.predicted_type,
            "gold_span": list(self.gold_span) if self.gold_span else None,
            "predicted_span": list(self.predicted_span) if self.predicted_span else None,
            "route": self.route, "section": self.section,
            "source": self.source, "provenance": self.provenance,
        }


class ExactMentionEvaluator:
    """Accumulates exact-match mention results plus an explanatory taxonomy."""

    def __init__(self) -> None:
        self.overall = _Counts()
        self.by_type: dict[str, _Counts] = {t: _Counts() for t in sorted(ENTITY_TYPES)}
        self.by_route: dict[str, _Counts] = {}
        self.by_section: dict[str, _Counts] = {}
        self.by_source: dict[str, _Counts] = {}
        self.by_length: dict[str, _Counts] = {b: _Counts() for b, _, _ in LENGTH_BUCKETS}
        self.by_provenance: dict[str, _Counts] = {}
        self.categories: dict[str, int] = dict.fromkeys(ERROR_CATEGORIES, 0)
        self.errors: list[ErrorRecord] = []
        self.examples = 0
        self.gold_total = 0
        self.predicted_total = 0

    # -- internal helpers ------------------------------------------------------
    @staticmethod
    def _bump(table: dict[str, _Counts], key: str, field_name: str) -> None:
        counts = table.setdefault(key, _Counts())
        setattr(counts, field_name, getattr(counts, field_name) + 1)

    def _credit(self, mention: Mention, source: str, field_name: str) -> None:
        setattr(self.overall, field_name, getattr(self.overall, field_name) + 1)
        self._bump(self.by_type, mention.entity_type, field_name)
        self._bump(self.by_route, mention.route or "unrouted", field_name)
        self._bump(self.by_section, mention.section or "unsectioned", field_name)
        self._bump(self.by_source, source or "unknown", field_name)
        self._bump(self.by_length, length_bucket(mention.length), field_name)
        self._bump(self.by_provenance, mention.provenance, field_name)

    # -- public API ------------------------------------------------------------
    def update(
        self,
        original_text: str,
        gold: Sequence[Mention],
        predicted: Sequence[Mention],
        *,
        example_id: str = "",
        source: str = "",
    ) -> None:
        """Score one example. Both sides are validated before anything is counted."""
        for mention in gold:
            validate_mention(mention, original_text, role="gold")
        for mention in predicted:
            validate_mention(mention, original_text, role="predicted")

        handle = privacy_safe_example_id(example_id)
        self.examples += 1
        self.gold_total += len(gold)
        self.predicted_total += len(predicted)

        gold_open = list(range(len(gold)))
        predicted_open = list(range(len(predicted)))

        # 1. Exact (start, end, type): the only true positives.
        gold_by_key: dict[tuple[int, int, str], list[int]] = {}
        for index in gold_open:
            gold_by_key.setdefault(gold[index].key, []).append(index)
        matched_gold: set[int] = set()
        matched_pred: set[int] = set()
        for index in predicted_open:
            bucket = gold_by_key.get(predicted[index].key)
            if bucket:
                gold_index = bucket.pop(0)
                matched_gold.add(gold_index)
                matched_pred.add(index)
                self.categories[EXACT_MATCH] += 1
                self._credit(predicted[index], source, "true_positive")

        remaining_gold = [i for i in gold_open if i not in matched_gold]
        remaining_pred = [i for i in predicted_open if i not in matched_pred]

        # 2. Same span, different type: FP for the predicted type AND FN for the
        #    gold type - the task's double penalty for a wrong type (spec §1).
        for pred_index in list(remaining_pred):
            prediction = predicted[pred_index]
            for gold_index in list(remaining_gold):
                reference = gold[gold_index]
                if (reference.start, reference.end) == (prediction.start, prediction.end):
                    remaining_gold.remove(gold_index)
                    remaining_pred.remove(pred_index)
                    self.categories[WRONG_TYPE] += 1
                    self._credit(prediction, source, "false_positive")
                    self._credit(reference, source, "false_negative")
                    self.errors.append(ErrorRecord(
                        handle, WRONG_TYPE, reference.entity_type, prediction.entity_type,
                        (reference.start, reference.end),
                        (prediction.start, prediction.end),
                        prediction.route, prediction.section, source, prediction.provenance))
                    break

        # 3. Overlapping, same type: a boundary error. Still FP + FN under exact
        #    matching; the category explains WHICH edge slipped.
        for pred_index in list(remaining_pred):
            prediction = predicted[pred_index]
            best: tuple[int, int] | None = None
            for gold_index in remaining_gold:
                reference = gold[gold_index]
                if (reference.entity_type == prediction.entity_type
                        and reference.overlaps(prediction)):
                    overlap = (min(reference.end, prediction.end)
                               - max(reference.start, prediction.start))
                    if best is None or overlap > best[1]:
                        best = (gold_index, overlap)
            if best is None:
                continue
            gold_index = best[0]
            reference = gold[gold_index]
            remaining_gold.remove(gold_index)
            remaining_pred.remove(pred_index)
            left_differs = reference.start != prediction.start
            right_differs = reference.end != prediction.end
            category = (BOTH_BOUNDARY if left_differs and right_differs
                        else LEFT_BOUNDARY if left_differs else RIGHT_BOUNDARY)
            self.categories[category] += 1
            self._credit(prediction, source, "false_positive")
            self._credit(reference, source, "false_negative")
            self.errors.append(ErrorRecord(
                handle, category, reference.entity_type, prediction.entity_type,
                (reference.start, reference.end), (prediction.start, prediction.end),
                prediction.route, prediction.section, source, prediction.provenance))

        # 4. Whatever is left over.
        for gold_index in remaining_gold:
            reference = gold[gold_index]
            self.categories[MISSED] += 1
            self._credit(reference, source, "false_negative")
            self.errors.append(ErrorRecord(
                handle, MISSED, reference.entity_type, "",
                (reference.start, reference.end), None,
                reference.route, reference.section, source, reference.provenance))
        for pred_index in remaining_pred:
            prediction = predicted[pred_index]
            # A spurious prediction that collides with an already-matched one is
            # reported as a duplicate/overlap rather than an independent miss.
            collides = any(
                prediction.overlaps(predicted[other]) for other in matched_pred)
            category = DUPLICATE_OVERLAP if collides else SPURIOUS
            self.categories[category] += 1
            self._credit(prediction, source, "false_positive")
            self.errors.append(ErrorRecord(
                handle, category, "", prediction.entity_type,
                None, (prediction.start, prediction.end),
                prediction.route, prediction.section, source, prediction.provenance))

    def report(self, *, config_hash: str = "", label: str = "") -> dict[str, Any]:
        """Machine-readable results. Groupings are sorted for deterministic output."""
        def grouped(table: Mapping[str, _Counts]) -> dict[str, Any]:
            return {key: table[key].as_dict() for key in sorted(table)}

        return {
            "evaluator": "exact_mention_v1",
            "label": label,
            "config_hash": config_hash,
            "scope": "mention span and type only",
            "reproduces_complete_organizer_score": False,
            "excluded_from_score": ["assertions (Jaccard, 30%)",
                                    "ICD/RxNorm candidates (Jaccard, 40%)"],
            "examples": self.examples,
            "gold_mentions": self.gold_total,
            "predicted_mentions": self.predicted_total,
            "micro": self.overall.as_dict(),
            "by_type": grouped(self.by_type),
            "by_route": grouped(self.by_route),
            "by_section": grouped(self.by_section),
            "by_source": grouped(self.by_source),
            "by_length_bucket": grouped(self.by_length),
            "by_provenance": grouped(self.by_provenance),
            "error_categories": dict(sorted(self.categories.items())),
            "errors": [record.as_dict() for record in self.errors],
        }


def config_hash_of(config: Mapping[str, Any]) -> str:
    """Deterministic hash of a resolved configuration, for report provenance."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_markdown(report: Mapping[str, Any]) -> str:
    """Compact human-readable report. Contains no clinical text."""
    micro = report["micro"]
    lines = [
        f"# Exact mention evaluation — {report.get('label') or 'unlabelled'}",
        "",
        "**Scope:** mention span and type only. This is **not** the complete "
        "organizer score — assertions and ICD/RxNorm candidate Jaccard are not "
        "integrated.",
        "",
        f"- config hash: `{report.get('config_hash') or 'n/a'}`",
        f"- examples: {report['examples']}   gold: {report['gold_mentions']}   "
        f"predicted: {report['predicted_mentions']}",
        "",
        "## Micro",
        "",
        "| precision | recall | F1 | TP | FP | FN |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| {micro['precision']:.4f} | {micro['recall']:.4f} | {micro['f1']:.4f} "
        f"| {micro['true_positive']} | {micro['false_positive']} "
        f"| {micro['false_negative']} |",
        "",
        "## Per type",
        "",
        "| type | precision | recall | F1 | TP | FP | FN |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, values in report["by_type"].items():
        lines.append(
            f"| {name} | {values['precision']:.4f} | {values['recall']:.4f} "
            f"| {values['f1']:.4f} | {values['true_positive']} "
            f"| {values['false_positive']} | {values['false_negative']} |")
    lines += ["", "## Error categories", "", "| category | count |", "| --- | --- |"]
    for name, count in report["error_categories"].items():
        lines.append(f"| {name} | {count} |")
    for section, key in (("Length bucket", "by_length_bucket"),
                         ("Provenance", "by_provenance"),
                         ("Source", "by_source")):
        lines += ["", f"## {section}", "", "| group | precision | recall | F1 |",
                  "| --- | --- | --- | --- |"]
        for name, values in report[key].items():
            lines.append(f"| {name} | {values['precision']:.4f} "
                         f"| {values['recall']:.4f} | {values['f1']:.4f} |")
    return "\n".join(lines) + "\n"


def evaluate_examples(
    examples: Iterable[Mapping[str, Any]],
    *,
    predictions: Mapping[str, Sequence[Mention]],
    label: str = "",
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate governed examples against per-example predicted mentions."""
    evaluator = ExactMentionEvaluator()
    for example in examples:
        example_id = str(example.get("example_id", ""))
        text = str(example.get("text", ""))
        gold = [
            Mention(int(entity["start"]), int(entity["end"]),
                    str(entity["target_type"]), str(entity["text"]))
            for entity in example.get("entities", []) or []
        ]
        evaluator.update(text, gold, predictions.get(example_id, ()),
                         example_id=example_id,
                         source=str(example.get("source_dataset", "")))
    return evaluator.report(
        config_hash=config_hash_of(config or {}), label=label)


__all__ = [
    "BOTH_BOUNDARY",
    "DUPLICATE_OVERLAP",
    "ERROR_CATEGORIES",
    "EXACT_MATCH",
    "LEFT_BOUNDARY",
    "LENGTH_BUCKETS",
    "MISSED",
    "PROVENANCE_DETERMINISTIC",
    "PROVENANCE_HYBRID",
    "PROVENANCE_NEURAL",
    "PROVENANCE_UNKNOWN",
    "RIGHT_BOUNDARY",
    "SPURIOUS",
    "WRONG_TYPE",
    "ErrorRecord",
    "ExactMentionError",
    "ExactMentionEvaluator",
    "Mention",
    "config_hash_of",
    "evaluate_examples",
    "length_bucket",
    "privacy_safe_example_id",
    "render_markdown",
    "validate_mention",
]
