"""L9 ontology-membership validation against the locked snapshots (spec §16).

Spec §16's fail-fast rule has two halves. The serialized offset/text half is owned
by :mod:`mednorm_vi.validator.final_gate`, which validates the final payload against
``original_text``. The snapshot-membership half is this module:

    "if ... a code is absent from the frozen KB snapshot, stop that file and write
     an audit log. Never silently repair output."

Until Audit 0053, L9 checked candidate *syntax* only (``validator/organizer.py``
verifies an RxCUI is numeric and leaves ICD-10 as an opaque string, with a comment
saying membership was "deferred until frozen KB snapshots exist"). Snapshots have
existed since Audit 0014. Membership was enforced inside the linker
(``linking/snapshot.py``), which is the wrong place for a *final* gate: the linker is
the component whose bug this check exists to catch, and a future retriever, reranker
or LLM selector could introduce a code without passing through it.

So this validator is deliberately **model-independent and retriever-independent**. It
takes the locked snapshots and the emitted output, and asks only whether each code is
in the snapshot its entity type requires. It never repairs, reorders or drops a code —
it reports, and L9's caller stops the file.

Appendix A's checklist item — "Every candidate exists in the frozen KB snapshot and
belongs to the correct ontology for its type" — is what this implements.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..schemas.constants import (
    CANDIDATE_ONTOLOGY_BY_TYPE,
    ORGANIZER_LABELS,
    TYPE_BY_ORGANIZER_LABEL,
)
from .results import Severity, ValidationIssue, ValidationResult

KB_MEMBERSHIP_VALIDATOR_VERSION = "l9-kb-membership-v1"

# Ontology identifiers as `schemas.constants.CANDIDATE_ONTOLOGY_BY_TYPE` names them.
ONTOLOGY_ICD10 = "ICD10"
ONTOLOGY_RXNORM = "RXNORM"

# The `index_type` each ontology's generated index declares.
INDEX_TYPE_BY_ONTOLOGY: Mapping[str, str] = {
    ONTOLOGY_ICD10: "icd10_vi",
    ONTOLOGY_RXNORM: "rxnorm",
}


class KbMembershipViolation(RuntimeError):
    """Raised by direct callers when membership-only validation fails.

    The validator itself only reports (see the module docstring). A caller that runs
    only the KB-membership validator may raise this exception. The canonical final
    packaging boundary raises :class:`mednorm_vi.validator.final_gate.FinalValidationError`
    because it aggregates organizer, offset, duplicate, ordering, snapshot and
    offered-set violations.
    """

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations = tuple(violations)
        super().__init__(
            "ontology membership validation failed; no output package was written: "
            + ", ".join(self.violations)
        )


class MembershipSource(Protocol):
    """Anything that can answer "is this code in the locked snapshot?".

    ``LocalIndex`` satisfies this, which is why no adapter is needed; a future
    snapshot reader that is not an index satisfies it too.

    The two members are read-only **properties** rather than plain annotations on
    purpose: a plain annotation declares a mutable attribute, which a frozen
    dataclass like ``LocalIndex`` cannot satisfy. This gate only ever reads them.
    """

    @property
    def index_type(self) -> str: ...

    @property
    def source_snapshot_id(self) -> str: ...

    def exists(self, concept_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class LockedSnapshots:
    """The snapshots L9 validates against. Absent means "cannot validate", not "ok"."""

    icd10: MembershipSource | None = None
    rxnorm: MembershipSource | None = None

    def source_for(self, ontology: str) -> MembershipSource | None:
        if ontology == ONTOLOGY_ICD10:
            return self.icd10
        if ontology == ONTOLOGY_RXNORM:
            return self.rxnorm
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "icd10_snapshot_id": (
                self.icd10.source_snapshot_id if self.icd10 is not None else None
            ),
            "rxnorm_snapshot_id": (
                self.rxnorm.source_snapshot_id if self.rxnorm is not None else None
            ),
        }


def _final_candidate_eligible(source: object, code: str) -> bool:
    """Whether the snapshot permits this code as a final candidate.

    Audit 0058's competition snapshot contains two populations: searchable concepts,
    which may be emitted, and closure-only graph endpoints, which exist so traversal
    has somewhere to land and may not. This gate reads the membership the snapshot
    states, without importing a linker — the linker is the component whose bug this
    check exists to catch.

    A snapshot that states no membership permits everything. Silence must not mean
    "closure-only": the legacy indices draw no such distinction, and reading their
    silence as exclusion would empty every candidate list.
    """
    checker = getattr(source, "final_candidate_eligible", None)
    if callable(checker):
        return bool(checker(code))
    records = getattr(source, "records", None)
    if isinstance(records, Mapping):
        record = records.get(code)
        if isinstance(record, Mapping):
            metadata = record.get("metadata")
            if isinstance(metadata, Mapping):
                membership = str(metadata.get("membership", ""))
                if membership:
                    return membership == "searchable"
    return True


def _issue(
    code: str, message: str, document_id: str | None, entity_index: int | None, **ctx: object
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        severity=Severity.ERROR,
        document_id=document_id,
        entity_index=entity_index,
        context=ctx,
    )


def validate_entity_candidates(
    entity: Mapping[str, Any],
    snapshots: LockedSnapshots,
    *,
    document_id: str | None = None,
    entity_index: int | None = None,
    offered_codes: Iterable[str] | None = None,
) -> ValidationResult:
    """Validate one organizer entity's candidate list against the locked snapshots.

    ``offered_codes``, when supplied, is the set the retrieval stage actually
    offered for this mention. A code outside it fails even if the snapshot contains
    it — spec P7 forbids a model *introducing* a code, and "it exists" is not the
    same as "it was retrieved for this mention".
    """
    issues: list[ValidationIssue] = []
    organizer_type = entity.get("type")
    if organizer_type not in ORGANIZER_LABELS:
        # `validator.organizer` owns the type vocabulary; membership cannot be
        # judged without a known type, so report and stop here.
        return ValidationResult.from_issues(
            [
                _issue(
                    "kb.unknown_type",
                    f"cannot validate candidates for unknown type {organizer_type!r}",
                    document_id,
                    entity_index,
                    value=organizer_type,
                )
            ]
        )

    internal_type = TYPE_BY_ORGANIZER_LABEL[organizer_type]
    expected_ontology = CANDIDATE_ONTOLOGY_BY_TYPE.get(internal_type)
    raw = entity.get("candidates")

    if expected_ontology is None:
        # SYMPTOM / TEST_NAME / TEST_RESULT carry no candidates at all (spec §7.3).
        if raw:
            issues.append(
                _issue(
                    "kb.candidates_on_type_without_ontology",
                    f"{organizer_type!r} takes no candidates but {len(list(raw))} were "
                    "emitted; cross-ontology linking is forbidden",
                    document_id,
                    entity_index,
                    entity_type=organizer_type,
                )
            )
        return ValidationResult.from_issues(issues)

    if raw is None:
        return ValidationResult.from_issues(issues)
    if not isinstance(raw, (list, tuple)):
        return ValidationResult.from_issues(
            [
                _issue(
                    "kb.candidates_not_list",
                    f"candidates must be a list, got {type(raw)!r}",
                    document_id,
                    entity_index,
                )
            ]
        )

    codes = [str(code) for code in raw]
    if not codes:
        # An entity that emitted no candidate makes no membership claim, so there is
        # nothing here that could be absent from a snapshot. Requiring a snapshot to
        # certify an empty list would reject a legitimate `deterministic`-mode run
        # that configures no KB at all — a false positive, not a stricter gate.
        # Missing *coverage* is a linker concern; this validator judges membership.
        return ValidationResult.from_issues(issues)

    # Duplicates: a repeated code inflates a Jaccard denominator for nothing.
    seen: set[str] = set()
    for code in codes:
        if code in seen:
            issues.append(
                _issue(
                    "kb.duplicate_candidate",
                    f"candidate {code!r} appears more than once",
                    document_id,
                    entity_index,
                    value=code,
                )
            )
        seen.add(code)

    # Deterministic ordering is a property of the *decoder*, not of a single
    # document, so it cannot be judged from one list. It is asserted by running the
    # pipeline twice and comparing (see tests); this validator deliberately does not
    # pretend to check it here.

    source = snapshots.source_for(expected_ontology)
    if source is None:
        issues.append(
            _issue(
                "kb.snapshot_unavailable",
                f"{organizer_type!r} requires the {expected_ontology} snapshot to "
                "validate its candidates, and none was supplied; membership cannot be "
                "confirmed, so the file is not accepted",
                document_id,
                entity_index,
                ontology=expected_ontology,
            )
        )
        return ValidationResult.from_issues(issues)

    expected_index_type = INDEX_TYPE_BY_ONTOLOGY.get(expected_ontology)
    if expected_index_type and source.index_type != expected_index_type:
        issues.append(
            _issue(
                "kb.wrong_snapshot",
                f"candidates for {organizer_type!r} were validated against index type "
                f"{source.index_type!r}, expected {expected_index_type!r}; a stale or "
                "wrong snapshot cannot certify membership",
                document_id,
                entity_index,
                ontology=expected_ontology,
                index_type=source.index_type,
            )
        )
        return ValidationResult.from_issues(issues)

    allowed = set(str(code) for code in offered_codes) if offered_codes is not None else None
    for code in codes:
        if not _final_candidate_eligible(source, code):
            issues.append(
                _issue(
                    "kb.candidate_not_final_eligible",
                    f"candidate {code!r} is in {expected_ontology} snapshot "
                    f"{source.source_snapshot_id!r} but is not a final-candidate-eligible "
                    "concept; a closure-only graph endpoint supports traversal and may "
                    "never be emitted (Audit 0058)",
                    document_id,
                    entity_index,
                    value=code,
                    snapshot_id=source.source_snapshot_id,
                )
            )
            continue
        if not source.exists(code):
            issues.append(
                _issue(
                    "kb.candidate_not_in_snapshot",
                    f"candidate {code!r} is not in {expected_ontology} snapshot "
                    f"{source.source_snapshot_id!r}",
                    document_id,
                    entity_index,
                    value=code,
                    snapshot_id=source.source_snapshot_id,
                )
            )
            continue
        if allowed is not None and code not in allowed:
            issues.append(
                _issue(
                    "kb.candidate_not_offered",
                    f"candidate {code!r} exists in the snapshot but was not offered for "
                    "this mention; a code may be selected, never introduced (spec P7)",
                    document_id,
                    entity_index,
                    value=code,
                )
            )
    return ValidationResult.from_issues(issues)


def validate_document_candidates(
    root: object,
    snapshots: LockedSnapshots,
    *,
    document_id: str | None = None,
    offered_codes_by_index: Mapping[int, Sequence[str]] | None = None,
) -> ValidationResult:
    """Validate every entity's candidates in one organizer document."""
    if not isinstance(root, list):
        return ValidationResult.from_issues(
            [
                ValidationIssue(
                    code="kb.root_not_list",
                    message=f"document root must be a JSON list, got {type(root)!r}",
                    document_id=document_id,
                )
            ]
        )
    result = ValidationResult()
    for index, entity in enumerate(root):
        if not isinstance(entity, Mapping):
            continue  # `validator.organizer` reports the shape error
        offered = (offered_codes_by_index or {}).get(index)
        result = result.merged_with(
            validate_entity_candidates(
                entity,
                snapshots,
                document_id=document_id,
                entity_index=index,
                offered_codes=offered,
            )
        )
    return result


@dataclass(frozen=True, slots=True)
class MembershipReport:
    """Summary for the run manifest. Carries no clinical text."""

    documents: int = 0
    entities: int = 0
    candidates: int = 0
    violations: tuple[str, ...] = field(default_factory=tuple)
    snapshots: Mapping[str, Any] = field(default_factory=dict)
    version: str = KB_MEMBERSHIP_VALIDATOR_VERSION

    @property
    def ok(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {
            "kb_membership_validator_version": self.version,
            "documents": self.documents,
            "entities": self.entities,
            "candidates": self.candidates,
            "violation_count": len(self.violations),
            "violations": list(self.violations),
            "snapshots": dict(self.snapshots),
            "ok": self.ok,
            "contains_clinical_text": False,
        }


__all__ = [
    "INDEX_TYPE_BY_ONTOLOGY",
    "KB_MEMBERSHIP_VALIDATOR_VERSION",
    "ONTOLOGY_ICD10",
    "ONTOLOGY_RXNORM",
    "KbMembershipViolation",
    "LockedSnapshots",
    "MembershipReport",
    "MembershipSource",
    "validate_document_candidates",
    "validate_entity_candidates",
]
