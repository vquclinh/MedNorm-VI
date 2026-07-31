"""Final-candidate eligibility gate over the competition candidate index (§8, §10).

The v3 snapshot contains two populations that a retriever can reach but only one that
may be emitted: 82,429 searchable RxNorm concepts and 129,520 closure-only endpoints
that exist purely so graph traversal has somewhere to land. The whole point of adding
the closure was to improve traversal *without* widening what the runtime may answer
with, and that separation is only real if something enforces it.

This module is that something. It reads ``competition_candidate_index_v1.csv`` and
answers one question per code — may this be emitted as a final organizer candidate? —
using the ``final_candidate_eligible`` column the builder wrote from the concept's
runtime role.

It is deliberately **retriever-independent and linker-independent**, for the same
reason :mod:`mednorm_vi.validator.kb_membership` is: the linker is the component whose
bug this gate exists to catch, so a gate that trusted the linker's own membership view
would catch nothing. It imports no linker, no index builder and no model.

``reason_for`` distinguishes the three ways a code can fail, because "not in the
snapshot" and "in the snapshot but not searchable" are different defects with
different fixes.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

CANDIDATE_ELIGIBILITY_VERSION = "competition-candidate-eligibility-v1"

#: Ontology identifiers as the candidate index writes them.
ONTOLOGY_ICD10 = "icd10_vi"
ONTOLOGY_RXNORM = "rxnorm"

#: Refusal reasons.
ELIGIBLE = "eligible"
NOT_IN_SNAPSHOT = "code_absent_from_snapshot"
NOT_FINAL_ELIGIBLE = "code_present_but_not_final_candidate_eligible"


@dataclass(frozen=True, slots=True)
class CandidateEligibility:
    """Which codes the competition snapshot permits as final organizer candidates."""

    snapshot_id: str
    eligible: frozenset[tuple[str, str]]
    known: frozenset[tuple[str, str]]
    roles: Mapping[tuple[str, str], str]
    version: str = CANDIDATE_ELIGIBILITY_VERSION

    def may_emit(self, ontology: str, code: str) -> bool:
        """Whether ``code`` may appear in a final candidate list for ``ontology``."""
        return (ontology, code) in self.eligible

    def reason_for(self, ontology: str, code: str) -> str:
        key = (ontology, code)
        if key in self.eligible:
            return ELIGIBLE
        if key in self.known:
            return NOT_FINAL_ELIGIBLE
        return NOT_IN_SNAPSHOT

    def runtime_role(self, ontology: str, code: str) -> str:
        return self.roles.get((ontology, code), "")

    def filter_candidates(
        self, ontology: str, codes: Iterable[str]
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        """Split codes into those that may be emitted and those refused, with reasons.

        Order is preserved on both sides: the caller's deterministic ordering must
        survive the gate.
        """
        kept: list[str] = []
        refused: list[tuple[str, str]] = []
        for code in codes:
            if self.may_emit(ontology, code):
                kept.append(code)
            else:
                refused.append((code, self.reason_for(ontology, code)))
        return tuple(kept), tuple(refused)


def load_candidate_eligibility(path: str | Path) -> CandidateEligibility:
    """Load the eligibility view from a competition candidate index CSV."""
    eligible: set[tuple[str, str]] = set()
    known: set[tuple[str, str]] = set()
    roles: dict[tuple[str, str], str] = {}
    snapshot_id = ""
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ontology = str(row.get("ontology", ""))
            code = str(row.get("code", ""))
            if not ontology or not code:
                continue
            key = (ontology, code)
            known.add(key)
            roles[key] = str(row.get("runtime_role", ""))
            if str(row.get("final_candidate_eligible", "")).lower() == "true":
                eligible.add(key)
            if not snapshot_id:
                snapshot_id = str(row.get("snapshot_id", ""))
    return CandidateEligibility(
        snapshot_id=snapshot_id,
        eligible=frozenset(eligible),
        known=frozenset(known),
        roles=roles,
    )


__all__ = [
    "CANDIDATE_ELIGIBILITY_VERSION",
    "ELIGIBLE",
    "NOT_FINAL_ELIGIBLE",
    "NOT_IN_SNAPSHOT",
    "ONTOLOGY_ICD10",
    "ONTOLOGY_RXNORM",
    "CandidateEligibility",
    "load_candidate_eligibility",
]
