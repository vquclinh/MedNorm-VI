"""ZS0 arm selection, organizer gate and package validation (Audit 0048).

Three responsibilities, all of them refusals by default:

* pick the submission arm by a **documented deterministic rule**, not by taste;
* let organizer inference run only when every precondition holds;
* validate the produced package before anyone uploads it.

The gate is the important part. Organizer inference writes the artifact that
gets submitted, so every way of reaching it accidentally is closed: the flag,
the exact authorization string, a selected arm, no trained expert active, no
training executed, a passing parameter budget, all 100 documents found exactly
once, and no external API configured.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas.constants import ORGANIZER_LABELS
from .ledger import LedgerReport
from .profile import ORGANIZER_AUTHORIZATION, ZS0_ARMS, ZS0ProfileError

SUBMISSION_CONTRACT_VERSION = "zs0-submission-v1"

EXPECTED_DOCUMENT_COUNT = 100
OUTPUT_FILENAMES: tuple[str, ...] = tuple(
    f"{index}.json" for index in range(1, EXPECTED_DOCUMENT_COUNT + 1))


class SubmissionError(RuntimeError):
    """Raised when a submission precondition or validation fails."""


# ---------------------------------------------------------------------------
# Deterministic arm selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One arm's measured result on the governed validation split."""

    arm: str
    exact_f1: float
    exact_precision: float
    exact_recall: float
    wrong_type_count: int
    malformed_proposal_count: int
    offset_violations: int
    runtime_seconds: float
    peak_vram_gib: float
    active_parameters: int

    def __post_init__(self) -> None:
        if self.arm not in ZS0_ARMS:
            raise ZS0ProfileError(f"unknown arm {self.arm!r}")

    @property
    def admissible(self) -> bool:
        """An arm with an offset violation is not a candidate at any score.

        A violated offset invariant means the submission would be malformed;
        there is no F1 that compensates for that.
        """
        return self.offset_violations == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm, "exact_f1": self.exact_f1,
            "exact_precision": self.exact_precision,
            "exact_recall": self.exact_recall,
            "wrong_type_count": self.wrong_type_count,
            "malformed_proposal_count": self.malformed_proposal_count,
            "offset_violations": self.offset_violations,
            "runtime_seconds": self.runtime_seconds,
            "peak_vram_gib": self.peak_vram_gib,
            "active_parameters": self.active_parameters,
            "admissible": self.admissible,
        }


# The rule, stated once and applied mechanically.
SELECTION_RULE = (
    "Among arms with zero offset violations, take the highest exact F1. Break a "
    "tie within 0.005 F1 by the fewest wrong-type errors, then the fewest "
    "malformed proposals, then the smallest active parameter total, then the "
    "simpler arm (A < B < C). The tolerance exists so a negligible F1 difference "
    "does not buy a larger model."
)
SELECTION_F1_TOLERANCE = 0.005


def select_arm(
    results: Sequence[ArmResult],
) -> tuple[ArmResult | None, dict[str, Any]]:
    """Apply :data:`SELECTION_RULE`. Returns ``(selected, report)``."""
    if not results:
        raise SubmissionError("no arm results were supplied")
    admissible = [r for r in results if r.admissible]
    report: dict[str, Any] = {
        "contract_version": SUBMISSION_CONTRACT_VERSION,
        "rule": SELECTION_RULE,
        "f1_tolerance": SELECTION_F1_TOLERANCE,
        "results": [r.as_dict() for r in results],
        "excluded_for_offset_violations": [
            r.arm for r in results if not r.admissible],
    }
    if not admissible:
        report["selected_arm"] = None
        report["reason"] = (
            "every arm violated the offset invariant; no submission may be built")
        return None, report

    best_f1 = max(r.exact_f1 for r in admissible)
    contenders = [r for r in admissible if best_f1 - r.exact_f1 <= SELECTION_F1_TOLERANCE]
    order = {arm: index for index, arm in enumerate(ZS0_ARMS)}
    selected = min(contenders, key=lambda r: (
        r.wrong_type_count, r.malformed_proposal_count, r.active_parameters,
        order[r.arm]))
    report["selected_arm"] = selected.arm
    report["contenders_within_tolerance"] = [r.arm for r in contenders]
    report["reason"] = (
        f"{selected.arm} has exact F1 {selected.exact_f1:.4f} (best {best_f1:.4f}, "
        f"within {SELECTION_F1_TOLERANCE}), {selected.wrong_type_count} wrong-type "
        f"errors, {selected.malformed_proposal_count} malformed proposals and "
        f"{selected.active_parameters:,} active parameters")
    return selected, report


# ---------------------------------------------------------------------------
# The organizer gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrganizerPreconditions:
    """Everything that must hold before organizer inference may run."""

    selected_arm: str | None
    active_trained_experts: tuple[str, ...]
    training_executed: bool
    ledger: LedgerReport | None
    local_gates_passed: bool
    discovered_documents: tuple[str, ...]
    external_api_configured: bool
    seed: int
    manifest_recorded: bool

    def failures(self) -> tuple[str, ...]:
        problems: list[str] = []
        if not self.selected_arm:
            problems.append("no ZS0 arm was selected")
        if self.active_trained_experts:
            problems.append(
                f"trained experts are active: {sorted(self.active_trained_experts)}")
        if self.training_executed:
            problems.append("training code was executed in this session")
        if self.ledger is None:
            problems.append("no parameter ledger was built")
        elif not self.ledger.within_budget:
            problems.append("the parameter budget did not pass")
        if not self.local_gates_passed:
            problems.append("a local schema/offset/ontology gate did not pass")
        unique = set(self.discovered_documents)
        if len(self.discovered_documents) != EXPECTED_DOCUMENT_COUNT:
            problems.append(
                f"discovered {len(self.discovered_documents)} organizer documents, "
                f"expected {EXPECTED_DOCUMENT_COUNT}")
        elif len(unique) != EXPECTED_DOCUMENT_COUNT:
            problems.append("an organizer document was discovered more than once")
        if self.external_api_configured:
            problems.append("an external model API is configured")
        if not self.manifest_recorded:
            problems.append("no run manifest was recorded")
        return tuple(problems)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_arm": self.selected_arm,
            "active_trained_experts": list(self.active_trained_experts),
            "training_executed": self.training_executed,
            "parameter_budget_passed": (
                self.ledger.within_budget if self.ledger is not None else False),
            "local_gates_passed": self.local_gates_passed,
            "discovered_documents": len(self.discovered_documents),
            "external_api_configured": self.external_api_configured,
            "seed": self.seed,
            "manifest_recorded": self.manifest_recorded,
            "failures": list(self.failures()),
        }


def assert_organizer_inference_allowed(
    *, enabled: bool, confirmation: str, preconditions: OrganizerPreconditions,
) -> None:
    """The whole gate. Any unmet condition refuses, by name."""
    if not enabled:
        raise SubmissionError(
            "organizer inference is disabled; RUN_ORGANIZER_INFERENCE ships False "
            "and an operator must enable it explicitly")
    if confirmation != ORGANIZER_AUTHORIZATION:
        raise SubmissionError(
            f"organizer inference requires the exact authorization string "
            f"{ORGANIZER_AUTHORIZATION!r}")
    failures = preconditions.failures()
    if failures:
        raise SubmissionError(
            "organizer inference refused; unmet preconditions: "
            + "; ".join(failures))


# ---------------------------------------------------------------------------
# Package validation
# ---------------------------------------------------------------------------


def validate_output_payload(
    name: str, payload: str, *, original_text: str,
    allowed_codes: Mapping[str, frozenset[str]] | None = None,
) -> tuple[str, ...]:
    """Validate one output JSON. Returns the problems found, by name."""
    problems: list[str] = []
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as error:
        return (f"{name}: invalid JSON ({error.msg})",)
    entities = document.get("entities") if isinstance(document, dict) else None
    if not isinstance(entities, list):
        return (f"{name}: missing an 'entities' list",)

    seen: set[tuple[Any, ...]] = set()
    previous: tuple[int, int] | None = None
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            problems.append(f"{name}[{index}]: entity is not an object")
            continue
        entity_type = entity.get("type")
        if entity_type not in ORGANIZER_LABELS:
            problems.append(f"{name}[{index}]: type {entity_type!r} is not allowed")
        start, end = entity.get("start_pos"), entity.get("end_pos")
        if not isinstance(start, int) or not isinstance(end, int):
            problems.append(f"{name}[{index}]: non-integer position")
            continue
        if end <= start:
            problems.append(f"{name}[{index}]: end_pos must exceed start_pos")
            continue
        if end > len(original_text):
            problems.append(f"{name}[{index}]: end_pos beyond the document")
            continue
        text = entity.get("text")
        if original_text[start:end] != text:
            problems.append(
                f"{name}[{index}]: text does not equal original_text[{start}:{end}]")
        key = (entity_type, start, end, text)
        if key in seen:
            problems.append(f"{name}[{index}]: duplicate entity record")
        seen.add(key)
        if previous is not None and (start, end) < previous:
            problems.append(f"{name}[{index}]: entities are not deterministically ordered")
        previous = (start, end)
        if allowed_codes is not None:
            for code in entity.get("candidates", []) or []:
                ontology_codes = allowed_codes.get(str(entity_type), frozenset())
                if str(code) not in ontology_codes:
                    problems.append(
                        f"{name}[{index}]: candidate {code!r} is not in the locked "
                        "ontology snapshot")
    return tuple(problems)


def validate_package(zip_path: str | Path) -> tuple[str, ...]:
    """The ZIP must contain exactly the 100 JSON files, at its root."""
    problems: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    if any("/" in name for name in names):
        problems.append("the archive contains a directory entry; files must be at root")
    if sorted(names) != sorted(OUTPUT_FILENAMES):
        missing = sorted(set(OUTPUT_FILENAMES) - set(names))
        extra = sorted(set(names) - set(OUTPUT_FILENAMES))
        if missing:
            problems.append(f"missing {len(missing)} files, first: {missing[:3]}")
        if extra:
            problems.append(f"unexpected entries: {extra[:3]}")
    return tuple(problems)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_hashes(output_dir: str | Path, zip_path: str | Path) -> dict[str, str]:
    """Every hash the audit needs, in a deterministic order."""
    root = Path(output_dir)
    hashes = {name: sha256_file(root / name)
              for name in OUTPUT_FILENAMES if (root / name).is_file()}
    hashes["output.zip"] = sha256_file(zip_path)
    return hashes


__all__ = [
    "EXPECTED_DOCUMENT_COUNT",
    "OUTPUT_FILENAMES",
    "SELECTION_F1_TOLERANCE",
    "SELECTION_RULE",
    "SUBMISSION_CONTRACT_VERSION",
    "ArmResult",
    "OrganizerPreconditions",
    "SubmissionError",
    "assert_organizer_inference_allowed",
    "package_hashes",
    "select_arm",
    "sha256_file",
    "validate_output_payload",
    "validate_package",
]
