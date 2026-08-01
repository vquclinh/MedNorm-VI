"""Linker-gold annotation records: schema, validation and status vocabulary (Audit 0065).

Audit 0063 §7.1 established that the governed corpora carry **no** ICD or RxNorm gold
codes, which made retrieval Recall@k and MRR undefined rather than merely unmeasured.
This module defines the record that fixes that: one human-adjudicated linking decision,
with the evidence the reviewer actually saw attached to it.

Two rules matter more than the rest and are enforced here rather than trusted:

* **Human gold and silver are never mixed.** `adjudication_status` is the discriminator,
  and :func:`is_human_gold` is the only thing a benchmark may count. A fuzzy top-1 or an
  LLM prediction is not gold, and there is no status that would let one pretend to be.
* **ICD identity stays dotless.** The dotted form is display and submission metadata
  (Audit 0061), so `code` is the KB key and `display_code` is decoration. A validator that
  accepted a dotted `code` would quietly break the offered-set contract.

Validation is strict on purpose: a half-filled annotation is worse than none, because it
looks like evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "linker_gold_v1"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "schemas" / "annotation" / "linker_gold_v1.schema.json"
)

# --- controlled vocabularies ---------------------------------------------------

STATUS_UNLABELLED = "UNLABELLED"
STATUS_HUMAN_GOLD = "HUMAN_GOLD"
STATUS_HUMAN_UNCERTAIN = "HUMAN_UNCERTAIN"
STATUS_SILVER_UNIQUE_EXACT = "SILVER_UNIQUE_EXACT"
STATUS_SECOND_REVIEW_PENDING = "SECOND_REVIEW_PENDING"
STATUS_SECOND_REVIEW_AGREED = "SECOND_REVIEW_AGREED"
STATUS_SECOND_REVIEW_DISAGREED = "SECOND_REVIEW_DISAGREED"

ADJUDICATION_STATUSES: tuple[str, ...] = (
    STATUS_UNLABELLED,
    STATUS_HUMAN_GOLD,
    STATUS_HUMAN_UNCERTAIN,
    STATUS_SILVER_UNIQUE_EXACT,
    STATUS_SECOND_REVIEW_PENDING,
    STATUS_SECOND_REVIEW_AGREED,
    STATUS_SECOND_REVIEW_DISAGREED,
)

#: The only statuses a benchmark may treat as ground truth. Silver is deliberately absent.
HUMAN_GOLD_STATUSES: frozenset[str] = frozenset({STATUS_HUMAN_GOLD, STATUS_SECOND_REVIEW_AGREED})

ONTOLOGY_ICD10 = "ICD10"
ONTOLOGY_RXNORM = "RXNORM"
LABEL_DIAGNOSIS = "CHẨN_ĐOÁN"
LABEL_MEDICATION = "THUỐC"

#: spec §7.3 forbids the cross-links; this is that rule as data.
ONTOLOGY_FOR_LABEL: dict[str, str] = {
    LABEL_DIAGNOSIS: ONTOLOGY_ICD10,
    LABEL_MEDICATION: ONTOLOGY_RXNORM,
}

RETRIEVAL_CHANNELS: tuple[str, ...] = ("exact", "exact_ascii", "char_ngram", "sparse", "graph")
CONFIDENCES: tuple[str, ...] = ("high", "medium", "low")

SPECIFICITY_CATEGORY = "category"
SPECIFICITY_SPECIFIC = "specific"
SPECIFICITY_OTHER = "other"
SPECIFICITY_UNSPECIFIED = "unspecified"

_ANNOTATION_ID = re.compile(r"^lg1-[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LinkerGoldError(ValueError):
    """A record violates the annotation contract."""


# --- helpers -------------------------------------------------------------------


def annotation_id(document_hash: str, start: int, end: int) -> str:
    """Deterministic id: the same mention always gets the same id.

    Derived rather than random so re-sampling a pack cannot silently create a second
    record for a mention that is already adjudicated.
    """
    payload = f"{document_hash}:{int(start)}:{int(end)}".encode()
    return f"lg1-{hashlib.sha256(payload).hexdigest()[:16]}"


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_icd_specificity(code: str) -> str:
    """category / other / unspecified / specific for a DOTLESS ICD code.

    Audit 0063 §7.3 measured 45.9% of top-1 codes as category-or-unspecified, so this
    distinction is a first-class field rather than something a later analysis re-derives.
    """
    bare = code.replace(".", "").strip().upper()
    if not bare:
        raise LinkerGoldError("empty ICD code")
    if len(bare) <= 3:
        return SPECIFICITY_CATEGORY
    fourth = bare[3]
    if fourth == "8":
        return SPECIFICITY_OTHER
    if fourth == "9":
        return SPECIFICITY_UNSPECIFIED
    return SPECIFICITY_SPECIFIC


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# --- validation ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    valid: bool
    problems: tuple[str, ...]

    def raise_for_problems(self) -> None:
        if self.problems:
            raise LinkerGoldError(
                f"{len(self.problems)} annotation problem(s): " + "; ".join(self.problems)
            )


def validate_record(
    record: Any, *, snapshot_codes: frozenset[str] | None = None
) -> ValidationOutcome:
    """Strict structural + semantic validation of one annotation record.

    ``snapshot_codes`` is the set of concept ids in the frozen KB snapshot. When supplied,
    every selected code must exist in it - a gold code that is not in the snapshot the
    system can retrieve from is not a usable target, it is a data-entry error.
    """
    problems: list[str] = []
    if not isinstance(record, dict):
        return ValidationOutcome(False, ("record is not a JSON object",))

    required = (
        "annotation_id",
        "source_dataset",
        "source_document_hash",
        "mention_start",
        "mention_end",
        "mention_text",
        "context_before",
        "context_after",
        "entity_type",
        "ontology",
        "candidate_snapshot_id",
        "offered_candidates",
        "selected_codes",
        "rejected_codes",
        "no_valid_candidate",
        "adjudication_status",
        "reviewer_id",
        "reviewer_confidence",
        "adjudication_notes",
        "provenance",
        "created_at",
        "updated_at",
    )
    for key in required:
        if key not in record:
            problems.append(f"missing required field {key!r}")
    if problems:
        return ValidationOutcome(False, tuple(problems))

    if not _ANNOTATION_ID.match(str(record["annotation_id"])):
        problems.append(f"annotation_id is malformed: {record['annotation_id']!r}")
    if not _SHA256.match(str(record["source_document_hash"])):
        problems.append("source_document_hash is not a SHA-256 hex digest")

    start, end = record["mention_start"], record["mention_end"]
    if not isinstance(start, int) or not isinstance(end, int) or end <= start or start < 0:
        problems.append(f"mention offsets are not a valid half-open span: [{start}, {end})")
    elif len(str(record["mention_text"])) != end - start:
        problems.append(
            f"mention_text length {len(str(record['mention_text']))} does not match the "
            f"span width {end - start}"
        )

    label = str(record["entity_type"])
    ontology = str(record["ontology"])
    expected = ONTOLOGY_FOR_LABEL.get(label)
    if expected is None:
        problems.append(f"entity_type {label!r} carries no candidates")
    elif ontology != expected:
        problems.append(
            f"{label} must link to {expected}, not {ontology!r} (spec §7.3 forbids the cross-links)"
        )

    status = str(record["adjudication_status"])
    if status not in ADJUDICATION_STATUSES:
        problems.append(f"unknown adjudication_status {status!r}")

    confidence = record["reviewer_confidence"]
    if confidence is not None and confidence not in CONFIDENCES:
        problems.append(f"unknown reviewer_confidence {confidence!r}")

    selected = record["selected_codes"]
    rejected = record["rejected_codes"]
    no_valid = record["no_valid_candidate"]
    if not isinstance(selected, list) or not isinstance(rejected, list):
        problems.append("selected_codes and rejected_codes must be arrays")
        return ValidationOutcome(False, tuple(problems))
    if len(set(selected)) != len(selected):
        problems.append("selected_codes contains duplicates")
    if not isinstance(no_valid, bool):
        problems.append("no_valid_candidate must be a boolean")
    elif no_valid and selected:
        problems.append(
            "no_valid_candidate and selected_codes are mutually exclusive; a record cannot "
            "both name a correct code and assert that none is correct"
        )

    overlap = set(selected) & set(rejected)
    if overlap:
        problems.append(f"codes appear in both selected and rejected: {sorted(overlap)}")

    if status in HUMAN_GOLD_STATUSES and not selected and not no_valid:
        problems.append(
            f"{status} carries no decision: either selected_codes or no_valid_candidate must be set"
        )
    if status == STATUS_UNLABELLED and (selected or rejected or no_valid):
        problems.append("UNLABELLED records must carry no decision")

    candidates = record["offered_candidates"]
    if not isinstance(candidates, list):
        problems.append("offered_candidates must be an array")
    else:
        if len(candidates) > 10:
            problems.append(
                f"offered_candidates exceeds the ten shown to reviewers: {len(candidates)}"
            )
        offered_codes = set()
        for index, candidate in enumerate(candidates):
            problems.extend(
                f"offered_candidates[{index}]: {issue}"
                for issue in _validate_candidate(candidate, ontology)
            )
            if isinstance(candidate, dict) and "code" in candidate:
                offered_codes.add(str(candidate["code"]))
        unknown = set(selected) - offered_codes
        if unknown and candidates:
            problems.append(
                f"selected codes were not among those offered: {sorted(unknown)} - a reviewer "
                "may only choose from the evidence they were shown"
            )
        unknown_rejected = set(rejected) - offered_codes
        if unknown_rejected and candidates:
            problems.append(f"rejected codes were not offered: {sorted(unknown_rejected)}")

    if ontology == ONTOLOGY_ICD10:
        dotted = [code for code in selected if "." in str(code)]
        if dotted:
            problems.append(
                f"ICD selected_codes must be DOTLESS internal identities, got {dotted} - the "
                "dotted form is display/submission metadata only (Audit 0061)"
            )

    if snapshot_codes is not None:
        missing = [code for code in selected if code not in snapshot_codes]
        if missing:
            problems.append(
                f"selected codes absent from snapshot {record['candidate_snapshot_id']!r}: "
                f"{sorted(missing)}"
            )

    provenance = record["provenance"]
    if not isinstance(provenance, dict):
        problems.append("provenance must be an object")
    else:
        for key in ("sampling_seed", "stratum", "pack_id", "linker_version"):
            if key not in provenance:
                problems.append(f"provenance is missing {key!r}")

    return ValidationOutcome(not problems, tuple(problems))


def _validate_candidate(candidate: Any, ontology: str) -> list[str]:
    problems: list[str] = []
    if not isinstance(candidate, dict):
        return ["is not an object"]
    for key in (
        "code",
        "canonical_name",
        "aliases",
        "linker_rank",
        "score",
        "retrieval_channels",
        "structure",
    ):
        if key not in candidate:
            problems.append(f"missing {key!r}")
    if problems:
        return problems

    if not isinstance(candidate["linker_rank"], int) or candidate["linker_rank"] < 1:
        problems.append("linker_rank must be a positive integer")
    for channel in candidate["retrieval_channels"]:
        if channel not in RETRIEVAL_CHANNELS:
            problems.append(f"unknown retrieval channel {channel!r}")

    structure = candidate["structure"]
    if not isinstance(structure, dict):
        return [*problems, "structure is not an object"]
    if structure.get("ontology") != ontology:
        problems.append(
            f"structure.ontology {structure.get('ontology')!r} does not match the record's "
            f"{ontology!r}"
        )
    if ontology == ONTOLOGY_ICD10:
        if "." in str(candidate["code"]):
            problems.append("ICD candidate code must be dotless; use display_code for the dot")
        if structure.get("specificity_class") not in (
            SPECIFICITY_CATEGORY,
            SPECIFICITY_SPECIFIC,
            SPECIFICITY_OTHER,
            SPECIFICITY_UNSPECIFIED,
        ):
            problems.append(f"unknown specificity_class {structure.get('specificity_class')!r}")
    elif ontology == ONTOLOGY_RXNORM and "tty" not in structure:
        problems.append("RxNorm candidates must record TTY")
    return problems


def is_human_gold(record: Any) -> bool:
    """Whether a benchmark may count this record as ground truth.

    Silver is excluded by construction: `SILVER_UNIQUE_EXACT` is not in
    :data:`HUMAN_GOLD_STATUSES`, so no caller can accidentally average the two together.
    """
    return (
        isinstance(record, dict)
        and str(record.get("adjudication_status", "")) in HUMAN_GOLD_STATUSES
    )


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL annotation file. One malformed line fails the whole read."""
    records: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LinkerGoldError(f"{path}:{number}: not valid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise LinkerGoldError(f"{path}:{number}: record is not an object")
        records.append(row)
    return records


def blank_record(
    *,
    document_text_hash: str,
    source_dataset: str,
    start: int,
    end: int,
    mention_text: str,
    context_before: str,
    context_after: str,
    entity_type: str,
    candidate_snapshot_id: str,
    offered_candidates: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """An UNLABELLED record: evidence assembled, no decision taken."""
    ontology = ONTOLOGY_FOR_LABEL.get(entity_type)
    if ontology is None:
        raise LinkerGoldError(f"entity_type {entity_type!r} carries no candidates")
    now = utc_now()
    return {
        "annotation_id": annotation_id(document_text_hash, start, end),
        "source_dataset": source_dataset,
        "source_document_hash": document_text_hash,
        "mention_start": int(start),
        "mention_end": int(end),
        "mention_text": mention_text,
        "context_before": context_before,
        "context_after": context_after,
        "entity_type": entity_type,
        "ontology": ontology,
        "candidate_snapshot_id": candidate_snapshot_id,
        "offered_candidates": offered_candidates,
        "selected_codes": [],
        "rejected_codes": [],
        "no_valid_candidate": False,
        "adjudication_status": STATUS_UNLABELLED,
        "reviewer_id": "",
        "reviewer_confidence": None,
        "adjudication_notes": "",
        "provenance": provenance,
        "created_at": now,
        "updated_at": now,
        "second_review": None,
    }


__all__ = [
    "ADJUDICATION_STATUSES",
    "CONFIDENCES",
    "HUMAN_GOLD_STATUSES",
    "LABEL_DIAGNOSIS",
    "LABEL_MEDICATION",
    "ONTOLOGY_FOR_LABEL",
    "ONTOLOGY_ICD10",
    "ONTOLOGY_RXNORM",
    "RETRIEVAL_CHANNELS",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "SPECIFICITY_CATEGORY",
    "SPECIFICITY_OTHER",
    "SPECIFICITY_SPECIFIC",
    "SPECIFICITY_UNSPECIFIED",
    "STATUS_HUMAN_GOLD",
    "STATUS_HUMAN_UNCERTAIN",
    "STATUS_SECOND_REVIEW_AGREED",
    "STATUS_SECOND_REVIEW_DISAGREED",
    "STATUS_SECOND_REVIEW_PENDING",
    "STATUS_SILVER_UNIQUE_EXACT",
    "STATUS_UNLABELLED",
    "LinkerGoldError",
    "ValidationOutcome",
    "annotation_id",
    "blank_record",
    "classify_icd_specificity",
    "document_hash",
    "is_human_gold",
    "load_records",
    "utc_now",
    "validate_record",
]
