"""Code-linking evaluation contract (spec §9, §10, §18) — **no gold data ships here**.

Every audit since 0051 has recorded the same blocker: the governed corpus contains
**zero ontology codes**. Audit 0053 §15.4 measured it precisely — all 406 gold entities
in the bounded validation slice carry `start`, `end`, `text`, `target_type` and
`mapping_status: MAP_EXACT`, and not one carries an ICD-10 code or an RxCUI. Candidates
are 40% of the organizer metric, so 40% of the metric is **unmeasurable**, and no amount
of code fixes that.

This module is the contract a future *human-checked* set must satisfy. It is
deliberately only a contract:

* it defines the record shapes, the metrics and the error taxonomy;
* it **refuses** to score when gold is absent, rather than returning zeros;
* it **refuses** an annotation whose adjudication status is not settled;
* it ships **no data**, and the tests use synthetic miniature records only.

What it must never become is a label generator. Running the linker over the corpus and
calling its output gold would produce a number that measures the linker against itself.
The milestone forbids it, and so does this module: there is no function here that takes
predictions and emits gold, and ``load_gold_records`` rejects any record whose
``adjudication`` is not ``ADJUDICATED_*``.

The human artifact still required is specified in ``REQUIRED_ANNOTATION_ARTIFACT``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CODE_LINKING_CONTRACT_VERSION = "code-linking-eval-v1"

# Ontologies a gold record may name. Anything else is a schema error, not a warning.
ONTOLOGY_ICD10 = "ICD10"
ONTOLOGY_RXNORM = "RXNORM"
ONTOLOGIES: frozenset[str] = frozenset({ONTOLOGY_ICD10, ONTOLOGY_RXNORM})

# Adjudication status. Only the ADJUDICATED_* values may be used as gold.
ADJUDICATED_SINGLE = "ADJUDICATED_SINGLE"        # one acceptable code
ADJUDICATED_MULTIPLE = "ADJUDICATED_MULTIPLE"    # several equally acceptable codes
ADJUDICATED_NONE = "ADJUDICATED_NONE"            # correctly links to NO code
PENDING = "PENDING"                              # not yet reviewed
DISPUTED = "DISPUTED"                            # reviewers disagreed
USABLE_AS_GOLD: frozenset[str] = frozenset(
    {ADJUDICATED_SINGLE, ADJUDICATED_MULTIPLE, ADJUDICATED_NONE})

# Error categories (spec §18.2's ablation vocabulary, extended for linking).
ERR_EXACT = "exact_candidate_set"
ERR_PARTIAL = "partial_candidate_overlap"
ERR_MISSED = "gold_code_missed"
ERR_BROADER = "broader_than_gold"
ERR_MORE_SPECIFIC = "more_specific_than_gold"
ERR_WRONG_ONTOLOGY = "wrong_ontology"
ERR_STALE_CODE = "stale_code_not_in_snapshot"
ERR_SPURIOUS = "spurious_candidates_where_gold_is_none"
ERR_NO_PREDICTION = "no_prediction_where_gold_exists"
LINKING_ERROR_CATEGORIES: tuple[str, ...] = (
    ERR_EXACT, ERR_PARTIAL, ERR_MISSED, ERR_BROADER, ERR_MORE_SPECIFIC,
    ERR_WRONG_ONTOLOGY, ERR_STALE_CODE, ERR_SPURIOUS, ERR_NO_PREDICTION,
)

# The sentinel every audit must print instead of a fabricated number.
UNMEASURABLE = "UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD"

# What a human annotation round must deliver before any of this can run. Written here
# so the requirement lives next to the code that consumes it.
REQUIRED_ANNOTATION_ARTIFACT: Mapping[str, Any] = {
    "artifact": "governed code-linking gold set",
    "format": "JSONL, one CodeLinkingGoldRecord per line",
    "minimum_scope": {
        "documents": "drawn ONLY from the governed validation split, never internal_test",
        "mentions": "every DIAGNOSIS and MEDICATION mention in the sampled documents",
        "rationale": (
            "partial mention coverage makes recall uninterpretable: a missing "
            "annotation is indistinguishable from a correctly empty one"),
    },
    "per_record_requirements": [
        "exact character offsets, end-exclusive, verified against the source text",
        "the ontology required by the mention's type (spec §7.3)",
        "the locked snapshot id the codes were chosen from",
        "zero, one or many acceptable codes — all three are legitimate answers",
        "adjudication status; PENDING and DISPUTED are NOT gold",
        "annotator ids and the adjudication rule used, for provenance",
    ],
    "process_requirements": [
        "at least two independent clinical annotators per mention",
        "an explicit adjudication step for disagreements",
        "inter-annotator agreement reported before the set is used",
        "split identity recorded by SHA-256, never by filename",
    ],
    "explicitly_forbidden": [
        "codes produced by this repository's linkers",
        "codes produced by any language model",
        "codes transferred from another corpus without clinical review",
        "any record whose adjudication status is PENDING or DISPUTED",
    ],
}


class GoldSetUnavailable(RuntimeError):
    """Raised instead of returning zeros when no usable gold exists.

    Refusing is the whole point. A scorer that returns 0.0 for an absent gold set
    produces a number that looks like a measurement, and that number ends up in a
    report.
    """


class GoldRecordInvalid(ValueError):
    """Raised when a record does not satisfy the strict schema."""


@dataclass(frozen=True, slots=True)
class CodeLinkingGoldRecord:
    """One human-adjudicated mention→codes annotation."""

    document_id: str
    start: int
    end: int
    mention_text: str
    entity_type: str
    ontology: str
    snapshot_id: str
    acceptable_codes: frozenset[str]
    adjudication: str
    annotators: tuple[str, ...] = field(default_factory=tuple)
    adjudication_rule: str = ""
    split_sha256: str = ""

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise GoldRecordInvalid(
                f"end must exceed start, got [{self.start}, {self.end})")
        if self.ontology not in ONTOLOGIES:
            raise GoldRecordInvalid(f"unknown ontology {self.ontology!r}")
        if self.adjudication not in USABLE_AS_GOLD | {PENDING, DISPUTED}:
            raise GoldRecordInvalid(
                f"unknown adjudication status {self.adjudication!r}")
        if not self.snapshot_id:
            raise GoldRecordInvalid("snapshot_id is required: a code is meaningless "
                                    "without the snapshot it was chosen from")
        if self.adjudication == ADJUDICATED_NONE and self.acceptable_codes:
            raise GoldRecordInvalid(
                "ADJUDICATED_NONE must carry an empty code set")
        if self.adjudication in {ADJUDICATED_SINGLE, ADJUDICATED_MULTIPLE} and (
                not self.acceptable_codes):
            raise GoldRecordInvalid(
                f"{self.adjudication} requires at least one acceptable code")
        if self.adjudication == ADJUDICATED_SINGLE and len(self.acceptable_codes) != 1:
            raise GoldRecordInvalid(
                "ADJUDICATED_SINGLE must carry exactly one code; use "
                "ADJUDICATED_MULTIPLE for several")

    @property
    def usable_as_gold(self) -> bool:
        return self.adjudication in USABLE_AS_GOLD

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.document_id, self.start, self.end)


@dataclass(frozen=True, slots=True)
class CodeLinkingPredictionRecord:
    """One system prediction for one mention, in the order the decoder emitted it."""

    document_id: str
    start: int
    end: int
    entity_type: str
    ontology: str
    snapshot_id: str
    predicted_codes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.document_id, self.start, self.end)


@dataclass(frozen=True, slots=True)
class RecallAtKResult:
    """Recall@K over mentions that have at least one acceptable gold code."""

    k: int
    hits: int
    eligible: int

    @property
    def recall(self) -> float:
        return self.hits / self.eligible if self.eligible else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k, "hits": self.hits, "eligible": self.eligible,
            "recall": round(self.recall, 6),
        }


@dataclass(frozen=True, slots=True)
class CandidateJaccardResult:
    """Mean Jaccard between predicted and acceptable code sets.

    ``ADJUDICATED_NONE`` records are included with Jaccard 1.0 when the system also
    predicted nothing — correctly withholding is a correct answer, and excluding those
    records would reward a system that always guesses.
    """

    mean_jaccard: float
    exact_set_matches: int
    scored: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_jaccard": round(self.mean_jaccard, 6),
            "exact_set_matches": self.exact_set_matches,
            "scored": self.scored,
        }


@dataclass(frozen=True, slots=True)
class LinkingErrorCategory:
    """One error bucket with its count and the mentions in it."""

    category: str
    count: int
    examples: tuple[tuple[str, int, int], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category, "count": self.count,
            "examples": [list(key) for key in self.examples],
        }


@dataclass(frozen=True, slots=True)
class CodeLinkingReport:
    """The full evaluation. Only constructible from usable gold."""

    contract_version: str = CODE_LINKING_CONTRACT_VERSION
    gold_records: int = 0
    predicted_records: int = 0
    recall_at_k: tuple[RecallAtKResult, ...] = field(default_factory=tuple)
    jaccard: CandidateJaccardResult | None = None
    error_categories: tuple[LinkingErrorCategory, ...] = field(default_factory=tuple)
    snapshot_ids: tuple[str, ...] = field(default_factory=tuple)
    split_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "code_linking_contract_version": self.contract_version,
            "gold_records": self.gold_records,
            "predicted_records": self.predicted_records,
            "recall_at_k": [r.as_dict() for r in self.recall_at_k],
            "candidate_jaccard": self.jaccard.as_dict() if self.jaccard else None,
            "error_categories": [c.as_dict() for c in self.error_categories],
            "snapshot_ids": list(self.snapshot_ids),
            "split_sha256": self.split_sha256,
        }


def gold_record_from_mapping(payload: Mapping[str, Any]) -> CodeLinkingGoldRecord:
    """Build one gold record from a JSON object, validating strictly."""
    missing = {"document_id", "start", "end", "ontology", "snapshot_id",
               "adjudication"} - set(payload)
    if missing:
        raise GoldRecordInvalid(f"missing required fields: {sorted(missing)}")
    return CodeLinkingGoldRecord(
        document_id=str(payload["document_id"]),
        start=int(payload["start"]), end=int(payload["end"]),
        mention_text=str(payload.get("mention_text", "")),
        entity_type=str(payload.get("entity_type", "")),
        ontology=str(payload["ontology"]),
        snapshot_id=str(payload["snapshot_id"]),
        acceptable_codes=frozenset(
            str(code) for code in payload.get("acceptable_codes", ()) or ()),
        adjudication=str(payload["adjudication"]),
        annotators=tuple(str(a) for a in payload.get("annotators", ()) or ()),
        adjudication_rule=str(payload.get("adjudication_rule", "")),
        split_sha256=str(payload.get("split_sha256", "")))


def load_gold_records(path: str | Path) -> tuple[CodeLinkingGoldRecord, ...]:
    """Load a gold JSONL file, refusing unsettled annotations.

    A ``PENDING`` or ``DISPUTED`` record raises rather than being skipped: silently
    dropping it would shrink the denominator and inflate every metric computed from it.
    """
    target = Path(path)
    if not target.is_file():
        raise GoldSetUnavailable(
            f"no code-linking gold set at {target}. {UNMEASURABLE}. "
            "See REQUIRED_ANNOTATION_ARTIFACT for what must be produced first.")
    records: list[CodeLinkingGoldRecord] = []
    with target.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = gold_record_from_mapping(json.loads(line))
            except (json.JSONDecodeError, GoldRecordInvalid) as exc:
                raise GoldRecordInvalid(f"{target}:{number}: {exc}") from exc
            if not record.usable_as_gold:
                raise GoldRecordInvalid(
                    f"{target}:{number}: adjudication {record.adjudication!r} is not "
                    "usable as gold; resolve it before evaluating")
            records.append(record)
    if not records:
        raise GoldSetUnavailable(f"{target} contains no gold records. {UNMEASURABLE}.")
    return tuple(records)


def recall_at_k(
    gold: Sequence[CodeLinkingGoldRecord],
    predictions: Sequence[CodeLinkingPredictionRecord],
    *,
    k: int,
) -> RecallAtKResult:
    """Recall@K: a hit when any acceptable code appears in the first ``k`` predictions."""
    by_key = {p.key: p for p in predictions}
    hits = 0
    eligible = 0
    for record in gold:
        if not record.acceptable_codes:
            continue  # ADJUDICATED_NONE has nothing to recall
        eligible += 1
        prediction = by_key.get(record.key)
        if prediction is None:
            continue
        if record.acceptable_codes & set(prediction.predicted_codes[:k]):
            hits += 1
    return RecallAtKResult(k=k, hits=hits, eligible=eligible)


def candidate_jaccard(
    gold: Sequence[CodeLinkingGoldRecord],
    predictions: Sequence[CodeLinkingPredictionRecord],
) -> CandidateJaccardResult:
    """Mean set Jaccard, counting a correctly-empty prediction as a perfect score."""
    by_key = {p.key: p for p in predictions}
    total = 0.0
    exact = 0
    scored = 0
    for record in gold:
        predicted = frozenset(
            by_key[record.key].predicted_codes) if record.key in by_key else frozenset()
        acceptable = record.acceptable_codes
        scored += 1
        if not acceptable and not predicted:
            total += 1.0
            exact += 1
            continue
        union = acceptable | predicted
        if not union:
            total += 1.0
            exact += 1
            continue
        overlap = len(acceptable & predicted) / len(union)
        total += overlap
        if predicted == acceptable:
            exact += 1
    return CandidateJaccardResult(
        mean_jaccard=total / scored if scored else 0.0,
        exact_set_matches=exact, scored=scored)


def categorize_errors(
    gold: Sequence[CodeLinkingGoldRecord],
    predictions: Sequence[CodeLinkingPredictionRecord],
    *,
    snapshot_members: Mapping[str, frozenset[str]] | None = None,
    ancestors_of: Mapping[str, frozenset[str]] | None = None,
) -> tuple[LinkingErrorCategory, ...]:
    """Bucket every gold mention into exactly one error category.

    ``ancestors_of`` maps a code to its ancestors, which is what distinguishes
    "broader than gold" from "simply wrong" — ICD-10's hierarchy makes that difference
    clinically meaningful. Without it, hierarchy-related errors fall into
    ``partial_candidate_overlap`` rather than being guessed at.
    """
    by_key = {p.key: p for p in predictions}
    members = snapshot_members or {}
    ancestors = ancestors_of or {}
    buckets: dict[str, list[tuple[str, int, int]]] = {
        category: [] for category in LINKING_ERROR_CATEGORIES}

    for record in gold:
        prediction = by_key.get(record.key)
        predicted = frozenset(prediction.predicted_codes) if prediction else frozenset()
        acceptable = record.acceptable_codes

        if prediction is not None and prediction.ontology != record.ontology:
            buckets[ERR_WRONG_ONTOLOGY].append(record.key)
            continue
        allowed = members.get(record.ontology)
        if allowed is not None and predicted - allowed:
            buckets[ERR_STALE_CODE].append(record.key)
            continue
        if not acceptable:
            (buckets[ERR_EXACT] if not predicted
             else buckets[ERR_SPURIOUS]).append(record.key)
            continue
        if not predicted:
            buckets[ERR_NO_PREDICTION].append(record.key)
            continue
        if predicted == acceptable:
            buckets[ERR_EXACT].append(record.key)
            continue
        if predicted & acceptable:
            buckets[ERR_PARTIAL].append(record.key)
            continue
        # No overlap: is the prediction an ancestor (broader) or a descendant?
        if any(code in ancestors.get(gold_code, frozenset())
               for gold_code in acceptable for code in predicted):
            buckets[ERR_BROADER].append(record.key)
        elif any(gold_code in ancestors.get(code, frozenset())
                 for gold_code in acceptable for code in predicted):
            buckets[ERR_MORE_SPECIFIC].append(record.key)
        else:
            buckets[ERR_MISSED].append(record.key)

    return tuple(
        LinkingErrorCategory(
            category=category, count=len(keys), examples=tuple(keys[:5]))
        for category, keys in buckets.items())


def evaluate_code_linking(
    gold: Sequence[CodeLinkingGoldRecord],
    predictions: Sequence[CodeLinkingPredictionRecord],
    *,
    ks: Iterable[int] = (1, 5, 10),
    snapshot_members: Mapping[str, frozenset[str]] | None = None,
    ancestors_of: Mapping[str, frozenset[str]] | None = None,
) -> CodeLinkingReport:
    """Score predictions against human-adjudicated gold. Refuses without gold."""
    usable = [record for record in gold if record.usable_as_gold]
    if not usable:
        raise GoldSetUnavailable(
            f"no usable gold records supplied. {UNMEASURABLE}. Recall@K and candidate "
            "Jaccard must not be reported as 0.0 in this situation.")
    return CodeLinkingReport(
        gold_records=len(usable),
        predicted_records=len(predictions),
        recall_at_k=tuple(
            recall_at_k(usable, predictions, k=k) for k in sorted(set(ks))),
        jaccard=candidate_jaccard(usable, predictions),
        error_categories=categorize_errors(
            usable, predictions, snapshot_members=snapshot_members,
            ancestors_of=ancestors_of),
        snapshot_ids=tuple(sorted({record.snapshot_id for record in usable})),
        split_sha256=next(
            (record.split_sha256 for record in usable if record.split_sha256), ""))


def candidate_quality_status(gold_path: str | Path | None = None) -> str:
    """What an audit must print for candidate quality today.

    Returns the ``UNMEASURABLE`` sentinel unless a real gold file exists. This is the
    function an audit should call rather than writing a number by hand.
    """
    if gold_path is None:
        return UNMEASURABLE
    try:
        records = load_gold_records(gold_path)
    except (GoldSetUnavailable, GoldRecordInvalid):
        return UNMEASURABLE
    return f"MEASURABLE:{len(records)}_adjudicated_records"


__all__ = [
    "ADJUDICATED_MULTIPLE",
    "ADJUDICATED_NONE",
    "ADJUDICATED_SINGLE",
    "CODE_LINKING_CONTRACT_VERSION",
    "DISPUTED",
    "ERR_BROADER",
    "ERR_EXACT",
    "ERR_MISSED",
    "ERR_MORE_SPECIFIC",
    "ERR_NO_PREDICTION",
    "ERR_PARTIAL",
    "ERR_SPURIOUS",
    "ERR_STALE_CODE",
    "ERR_WRONG_ONTOLOGY",
    "LINKING_ERROR_CATEGORIES",
    "ONTOLOGIES",
    "ONTOLOGY_ICD10",
    "ONTOLOGY_RXNORM",
    "PENDING",
    "REQUIRED_ANNOTATION_ARTIFACT",
    "UNMEASURABLE",
    "USABLE_AS_GOLD",
    "CandidateJaccardResult",
    "CodeLinkingGoldRecord",
    "CodeLinkingPredictionRecord",
    "CodeLinkingReport",
    "GoldRecordInvalid",
    "GoldSetUnavailable",
    "LinkingErrorCategory",
    "RecallAtKResult",
    "candidate_jaccard",
    "candidate_quality_status",
    "categorize_errors",
    "evaluate_code_linking",
    "gold_record_from_mapping",
    "load_gold_records",
    "recall_at_k",
]
