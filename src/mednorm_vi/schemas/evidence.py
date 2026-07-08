"""L5 Specialists evidence contracts (spec sections 8-10).

Specialists produce SCORED evidence, never final answers. An ``EvidenceBundle``
gathers assertion and candidate evidence for one hypothesis. Candidates are
ontology-scoped (ICD-10 for DIAGNOSIS, RxNorm for MEDICATION) and must originate
from the frozen KB snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AssertionEvidence:
    """Evidence for a single assertion label on an entity (multi-label).

    ``probability`` is the calibrated (or pre-calibration) score that the label
    applies. Contributing cues/scope/section are recorded for provenance.
    """

    label: str  # one of ASSERTION_LABELS
    probability: float
    cue_texts: tuple[str, ...] = field(default_factory=tuple)
    cue_spans: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    section_id: str | None = None
    scope_source: str | None = None  # e.g. "A3-scope-resolver", "A1-section-prior"


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """One retrieved ontology candidate with its provenance and scores.

    ``code`` is an ICD-10 code or an RxCUI. ``ontology`` names which one. The
    code MUST exist in the frozen KB snapshot (enforced later by the validator /
    linker, not by this dataclass).
    """

    code: str
    ontology: str  # "ICD10" | "RXNORM"
    score: float
    canonical_name: str | None = None
    retrieval_sources: tuple[str, ...] = field(default_factory=tuple)
    # Hierarchy/graph provenance (e.g. parent/child of a retrieved node).
    graph_provenance: dict[str, str] = field(default_factory=dict)
    membership_probability: float | None = None  # P(code in gold set)


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """All specialist evidence attached to one hypothesis."""

    hypothesis_id: str
    assertion_evidence: tuple[AssertionEvidence, ...] = field(default_factory=tuple)
    candidate_evidence: tuple[CandidateEvidence, ...] = field(default_factory=tuple)
    # Scored boundary feedback from linkers (they may NOT edit spans directly).
    boundary_feedback: dict[str, float] = field(default_factory=dict)
