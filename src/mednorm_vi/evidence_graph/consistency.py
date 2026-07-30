"""Deterministic graph-consistency contract (L6, spec §11).

Audit 0053 §7 reported that "consistency violations" was **not a measurable quantity**
in this repository, and refused to report it as zero. This module is what makes it
measurable — and the refusal was the right call, because the honest answer to "how many
violations?" was "there is no checker".

The design rule that matters most: **uncertainty is never encoded as false.** Every
check returns one of four verdicts —

```text
SUPPORTED       the evidence affirms the thing
CONTRADICTED    the evidence denies the thing
UNRESOLVED      the check applies but the evidence cannot settle it
NOT_APPLICABLE  the check does not apply to this subject at all
```

— because collapsing UNRESOLVED into CONTRADICTED invents contradictions, and
collapsing it into SUPPORTED hides them. Both are worse than saying "unknown", and the
downstream consumers (L7 escalation, L8 retention) treat the four differently on
purpose: only CONTRADICTED with ``fatal=True`` may block emission, while UNRESOLVED is
exactly the signal that should raise an escalation.

The report carries **no clinical text**: subjects are hypothesis/candidate/edge ids,
and evidence is expressed as ids, codes, coordinates and rule names. A test asserts
this against real fixtures.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..linking.models import LinkerResult
from ..resolution.models import EntityHypothesis
from ..schemas.constants import CANDIDATE_ONTOLOGY_BY_TYPE, TYPE_BY_ORGANIZER_LABEL
from ..specialists.assertion import AssertionDecision
from .graph import (
    REL_HAS_CANDIDATE,
    REL_HAS_RESULT,
    REL_IN_SECTION,
    REL_OVERLAPS,
    REL_SAME_SURFACE,
    REL_TREATS,
    ClinicalEvidenceGraph,
)

CONSISTENCY_VERSION = "graph-consistency-v1"

# The four verdicts. Never fold UNRESOLVED into either extreme.
SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
UNRESOLVED = "UNRESOLVED"
NOT_APPLICABLE = "NOT_APPLICABLE"
VERDICTS: tuple[str, ...] = (SUPPORTED, CONTRADICTED, UNRESOLVED, NOT_APPLICABLE)

# Downstream recommendations. Advice, not orders — L7/L8 decide.
REC_EMIT = "EMIT"
REC_EMIT_WITH_CAUTION = "EMIT_WITH_CAUTION"
REC_ESCALATE = "ESCALATE"
REC_WITHHOLD = "WITHHOLD"

# Rule identifiers, versioned so a report can be read years later.
RULE_SECTION_COMPAT = "C01.section_compatibility"
RULE_LAB_PAIR = "C02.test_pair_completeness"
RULE_ASSERTION_TYPE = "C03.assertion_entity_type_compatibility"
RULE_CANDIDATE_TYPE = "C04.candidate_entity_type_compatibility"
RULE_ICD_HIERARCHY = "C05.icd_hierarchy_compatibility"
RULE_RXNORM_STRUCTURED = "C06.rxnorm_structured_compatibility"
RULE_OVERLAP = "C07.overlap_competition"
RULE_DUPLICATE_CANDIDATE = "C08.duplicate_candidate_codes"
RULE_REPEATED_MENTION = "C09.repeated_mention_agreement"
RULE_ASSERTION_CONFLICT = "C10.conflicting_assertion_evidence"
RULE_MED_CONFLICT = "C11.unresolved_structured_medication_conflict"
RULE_UNSAFE_TREATS = "C12.unsafe_or_unsupported_treats"

RULES: tuple[str, ...] = (
    RULE_SECTION_COMPAT, RULE_LAB_PAIR, RULE_ASSERTION_TYPE, RULE_CANDIDATE_TYPE,
    RULE_ICD_HIERARCHY, RULE_RXNORM_STRUCTURED, RULE_OVERLAP,
    RULE_DUPLICATE_CANDIDATE, RULE_REPEATED_MENTION, RULE_ASSERTION_CONFLICT,
    RULE_MED_CONFLICT, RULE_UNSAFE_TREATS,
)

# Types that may carry assertions at all (spec §8.1) and the ontology per type (§7.3).
ASSERTION_ELIGIBLE: frozenset[str] = frozenset(
    {"CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "THUỐC"})
# Section categories incompatible with a *present-tense* clinical claim. Section
# evidence lowers confidence and can raise an escalation; on its own it never blocks.
HISTORY_SECTIONS: frozenset[str] = frozenset(
    {"tien_su", "tiensu", "history", "family_history", "tien_su_gia_dinh"})


@dataclass(frozen=True, slots=True)
class SupportSignal:
    """One piece of evidence for or against a check. Ids and values only, no text."""

    kind: str
    value: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True, slots=True)
class ConsistencyDecision:
    """The verdict of one rule applied to one subject."""

    rule: str
    subject_id: str
    verdict: str
    supporting: tuple[SupportSignal, ...] = field(default_factory=tuple)
    blocking: tuple[SupportSignal, ...] = field(default_factory=tuple)
    version: str = CONSISTENCY_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule, "subject_id": self.subject_id,
            "verdict": self.verdict,
            "supporting": [s.as_dict() for s in self.supporting],
            "blocking": [s.as_dict() for s in self.blocking],
            "rule_version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ConsistencyIssue:
    """A decision worth acting on, with an explicit fatal/advisory disposition."""

    rule: str
    subject_id: str
    verdict: str
    fatal: bool
    recommendation: str
    hypothesis_ids: tuple[str, ...] = field(default_factory=tuple)
    candidate_codes: tuple[str, ...] = field(default_factory=tuple)
    edge_keys: tuple[str, ...] = field(default_factory=tuple)
    supporting: tuple[SupportSignal, ...] = field(default_factory=tuple)
    blocking: tuple[SupportSignal, ...] = field(default_factory=tuple)
    version: str = CONSISTENCY_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule, "rule_version": self.version,
            "subject_id": self.subject_id, "verdict": self.verdict,
            "fatal": self.fatal, "recommendation": self.recommendation,
            "hypothesis_ids": list(self.hypothesis_ids),
            "candidate_codes": list(self.candidate_codes),
            "edge_keys": list(self.edge_keys),
            "supporting": [s.as_dict() for s in self.supporting],
            "blocking": [s.as_dict() for s in self.blocking],
        }


@dataclass(frozen=True, slots=True)
class GraphConsistencyReport:
    """The typed public L6 contract. L7 and L8 consume this, never the raw graph."""

    document_id: str
    decisions: tuple[ConsistencyDecision, ...] = field(default_factory=tuple)
    issues: tuple[ConsistencyIssue, ...] = field(default_factory=tuple)
    graph_hash: str = ""
    version: str = CONSISTENCY_VERSION

    # --- typed accessors L7/L8 use instead of poking at dictionaries ---------
    @property
    def fatal_issues(self) -> tuple[ConsistencyIssue, ...]:
        return tuple(i for i in self.issues if i.fatal)

    @property
    def advisory_issues(self) -> tuple[ConsistencyIssue, ...]:
        return tuple(i for i in self.issues if not i.fatal)

    def issues_for(self, hypothesis_id: str) -> tuple[ConsistencyIssue, ...]:
        return tuple(i for i in self.issues if hypothesis_id in i.hypothesis_ids)

    def has_fatal(self, hypothesis_id: str) -> bool:
        return any(i.fatal for i in self.issues_for(hypothesis_id))

    def blocked_candidate_codes(self, hypothesis_id: str) -> frozenset[str]:
        """Codes a fatal issue forbids emitting for this hypothesis."""
        return frozenset(
            code for issue in self.issues_for(hypothesis_id) if issue.fatal
            for code in issue.candidate_codes)

    def verdict_for(self, rule: str, subject_id: str) -> str:
        for decision in self.decisions:
            if decision.rule == rule and decision.subject_id == subject_id:
                return decision.verdict
        return NOT_APPLICABLE

    def decision_counts(self) -> dict[str, int]:
        counts = {verdict: 0 for verdict in VERDICTS}
        for decision in self.decisions:
            counts[decision.verdict] = counts.get(decision.verdict, 0) + 1
        return counts

    def issue_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {rule: 0 for rule in RULES}
        for issue in self.issues:
            counts[issue.rule] = counts.get(issue.rule, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        """Deterministic serialization. Carries no clinical text."""
        return {
            "consistency_version": self.version,
            "document_id": self.document_id,
            "graph_hash": self.graph_hash,
            "decision_counts": self.decision_counts(),
            "issue_counts": self.issue_counts(),
            "fatal_issues": len(self.fatal_issues),
            "advisory_issues": len(self.advisory_issues),
            "decisions": [d.as_dict() for d in self.decisions],
            "issues": [i.as_dict() for i in self.issues],
            "contains_clinical_text": False,
        }

    @property
    def report_hash(self) -> str:
        return hashlib.sha256(json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _signal(kind: str, value: object) -> SupportSignal:
    return SupportSignal(kind=kind, value=str(value))


def ontology_for(entity_type: str) -> str | None:
    """Ontology for an entity type, or ``None`` when the type takes no candidates.

    ``EntityHypothesis.entity_type`` holds the **organizer-facing** Vietnamese label
    (`THUỐC`), while ``CANDIDATE_ONTOLOGY_BY_TYPE`` is keyed by the internal English
    name (`MEDICATION`). Looking the Vietnamese label up directly returns ``None`` for
    every type, which reads as "this type takes no candidates" and turns every linked
    medication into a fatal contradiction. That defect withheld 100% of output before a
    test caught it, so the translation lives in one named function.
    """
    return CANDIDATE_ONTOLOGY_BY_TYPE.get(
        TYPE_BY_ORGANIZER_LABEL.get(entity_type, entity_type))


def evaluate_consistency(
    document_id: str,
    graph: ClinicalEvidenceGraph,
    hypotheses: Sequence[EntityHypothesis],
    assertions: Sequence[AssertionDecision] = (),
    link_results: Sequence[LinkerResult] = (),
    *,
    section_categories: Mapping[str, str] | None = None,
) -> GraphConsistencyReport:
    """Run every deterministic consistency check over one document's evidence."""
    decisions: list[ConsistencyDecision] = []
    issues: list[ConsistencyIssue] = []
    by_id = {h.hypothesis_id: h for h in hypotheses}
    assertion_by_id = {a.hypothesis_id: a for a in assertions}
    links_by_id = {r.mention_id: r for r in link_results}
    categories = section_categories or {}

    def record(
        rule: str, subject: str, verdict: str,
        supporting: Sequence[SupportSignal] = (),
        blocking: Sequence[SupportSignal] = (),
        *, fatal: bool = False, recommendation: str = REC_EMIT,
        hypothesis_ids: Sequence[str] = (), candidate_codes: Sequence[str] = (),
        edge_keys: Sequence[str] = (),
    ) -> None:
        decisions.append(ConsistencyDecision(
            rule, subject, verdict, tuple(supporting), tuple(blocking)))
        if verdict in {CONTRADICTED, UNRESOLVED}:
            issues.append(ConsistencyIssue(
                rule=rule, subject_id=subject, verdict=verdict, fatal=fatal,
                recommendation=recommendation,
                hypothesis_ids=tuple(hypothesis_ids) or (subject,),
                candidate_codes=tuple(candidate_codes), edge_keys=tuple(edge_keys),
                supporting=tuple(supporting), blocking=tuple(blocking)))

    # --- C01 section compatibility -------------------------------------------
    for h in hypotheses:
        sections = graph.neighbours(h.hypothesis_id, REL_IN_SECTION)
        if not sections:
            record(RULE_SECTION_COMPAT, h.hypothesis_id, NOT_APPLICABLE,
                   [_signal("no_section_evidence", "l1")])
            continue
        category = categories.get(sections[0].removeprefix("section:"), "")
        assertion = assertion_by_id.get(h.hypothesis_id)
        historical = assertion is not None and "isHistorical" in assertion.labels
        if category.casefold() in HISTORY_SECTIONS and not historical:
            # The section says history; the assertion layer did not. Neither is proof,
            # so this is UNRESOLVED, not a contradiction, and it is advisory.
            record(RULE_SECTION_COMPAT, h.hypothesis_id, UNRESOLVED,
                   [_signal("section", sections[0])],
                   [_signal("section_category", category),
                    _signal("assertion_labels", ",".join(
                        assertion.labels if assertion else ()))],
                   recommendation=REC_ESCALATE)
        else:
            record(RULE_SECTION_COMPAT, h.hypothesis_id, SUPPORTED,
                   [_signal("section", sections[0]), _signal("category", category)])

    # --- C02 TEST_NAME / TEST_RESULT pair completeness ------------------------
    for h in hypotheses:
        if h.entity_type not in {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}:
            record(RULE_LAB_PAIR, h.hypothesis_id, NOT_APPLICABLE)
            continue
        paired = bool(
            graph.neighbours(h.hypothesis_id, REL_HAS_RESULT)
            or any(e.target_id == h.hypothesis_id
                   for e in graph.edges_of(REL_HAS_RESULT)))
        if paired:
            record(RULE_LAB_PAIR, h.hypothesis_id, SUPPORTED,
                   [_signal("has_result_edge", "present")])
        elif h.has_result_pair_group_ids:
            # E2 recorded a pairing that produced no edge — a genuine contradiction
            # between two upstream contracts.
            record(RULE_LAB_PAIR, h.hypothesis_id, CONTRADICTED,
                   blocking=[_signal("pair_groups", ",".join(
                       h.has_result_pair_group_ids)),
                       _signal("has_result_edges", "absent")],
                   recommendation=REC_ESCALATE)
        else:
            record(RULE_LAB_PAIR, h.hypothesis_id, UNRESOLVED,
                   blocking=[_signal("unpaired", h.entity_type)],
                   recommendation=REC_EMIT_WITH_CAUTION)

    # --- C03 assertion / entity-type compatibility ---------------------------
    for hypothesis_id, assertion in sorted(assertion_by_id.items()):
        subject_h = by_id.get(hypothesis_id)
        if subject_h is None:
            record(RULE_ASSERTION_TYPE, hypothesis_id, CONTRADICTED,
                   blocking=[_signal("orphan_assertion", hypothesis_id)],
                   fatal=True, recommendation=REC_WITHHOLD)
            continue
        if not assertion.labels:
            record(RULE_ASSERTION_TYPE, hypothesis_id, NOT_APPLICABLE)
            continue
        if subject_h.entity_type in ASSERTION_ELIGIBLE:
            record(RULE_ASSERTION_TYPE, hypothesis_id, SUPPORTED,
                   [_signal("entity_type", subject_h.entity_type),
                    _signal("labels", ",".join(assertion.labels))])
        else:
            record(RULE_ASSERTION_TYPE, hypothesis_id, CONTRADICTED,
                   blocking=[_signal("entity_type", subject_h.entity_type),
                             _signal("labels", ",".join(assertion.labels))],
                   fatal=True, recommendation=REC_WITHHOLD)

    # --- C04 candidate / entity-type compatibility (spec §7.3) ---------------
    for hypothesis_id, result in sorted(links_by_id.items()):
        subject_h = by_id.get(hypothesis_id)
        if subject_h is None:
            continue
        ontology = ontology_for(subject_h.entity_type)
        if ontology is None and result.candidates:
            record(RULE_CANDIDATE_TYPE, hypothesis_id, CONTRADICTED,
                   blocking=[_signal("entity_type", subject_h.entity_type),
                             _signal("candidate_count", len(result.candidates))],
                   fatal=True, recommendation=REC_WITHHOLD,
                   candidate_codes=[c.code for c in result.candidates])
        elif ontology is None:
            record(RULE_CANDIDATE_TYPE, hypothesis_id, NOT_APPLICABLE)
        elif result.candidates:
            record(RULE_CANDIDATE_TYPE, hypothesis_id, SUPPORTED,
                   [_signal("ontology", ontology),
                    _signal("candidate_count", len(result.candidates))])
        else:
            record(RULE_CANDIDATE_TYPE, hypothesis_id, UNRESOLVED,
                   blocking=[_signal("ontology", ontology),
                             _signal("candidate_count", 0)],
                   recommendation=REC_EMIT_WITH_CAUTION)

    # --- C05 ICD hierarchy / C06 RxNorm structured compatibility -------------
    # Both read the linker's own evidence strings, which the Audit-0054 linkers
    # populate with tier, relationship and matched-field records.
    for hypothesis_id, result in sorted(links_by_id.items()):
        subject_h = by_id.get(hypothesis_id)
        if subject_h is None:
            continue
        rule = RULE_ICD_HIERARCHY if subject_h.entity_type == "CHẨN_ĐOÁN" else (
            RULE_RXNORM_STRUCTURED if subject_h.entity_type == "THUỐC" else "")
        if not rule:
            continue
        evidence = [e for c in result.candidates for e in c.evidence]
        supported = [e for e in evidence if e.startswith(
            ("tier:exact", "tier:supported", "tier:structured", "matched:",
             "supported_detail:"))]
        weak_only = result.candidates and not supported
        if not result.candidates:
            record(rule, hypothesis_id, NOT_APPLICABLE)
        elif weak_only:
            record(rule, hypothesis_id, UNRESOLVED,
                   blocking=[_signal("evidence", "lexical_only")],
                   recommendation=REC_ESCALATE,
                   candidate_codes=[c.code for c in result.candidates])
        else:
            record(rule, hypothesis_id, SUPPORTED,
                   [_signal("evidence", value) for value in sorted(set(supported))[:6]])

    # --- C07 overlap competition ---------------------------------------------
    for edge in graph.edges_of(REL_OVERLAPS):
        same_type = any(
            p == "same_type:True" for p in edge.provenance)
        subject = f"{edge.source_id}|{edge.target_id}"
        if same_type:
            # Two same-type hypotheses on overlapping coordinates: L4 should have
            # resolved one away. Both surviving is a contradiction.
            record(RULE_OVERLAP, subject, CONTRADICTED,
                   blocking=[_signal("overlap", "same_type"), *(
                       _signal("interval", p) for p in edge.provenance[:2])],
                   fatal=True, recommendation=REC_WITHHOLD,
                   hypothesis_ids=[edge.source_id, edge.target_id],
                   edge_keys=[f"{edge.relation}:{subject}"])
        else:
            record(RULE_OVERLAP, subject, UNRESOLVED,
                   blocking=[_signal("overlap", "cross_type")],
                   recommendation=REC_EMIT_WITH_CAUTION,
                   hypothesis_ids=[edge.source_id, edge.target_id],
                   edge_keys=[f"{edge.relation}:{subject}"])

    # --- C08 duplicate candidate codes ---------------------------------------
    for hypothesis_id, result in sorted(links_by_id.items()):
        codes = [c.code for c in result.candidates]
        duplicates = sorted({code for code in codes if codes.count(code) > 1})
        if not codes:
            record(RULE_DUPLICATE_CANDIDATE, hypothesis_id, NOT_APPLICABLE)
        elif duplicates:
            record(RULE_DUPLICATE_CANDIDATE, hypothesis_id, CONTRADICTED,
                   blocking=[_signal("duplicate", code) for code in duplicates],
                   fatal=True, recommendation=REC_WITHHOLD,
                   candidate_codes=duplicates)
        else:
            record(RULE_DUPLICATE_CANDIDATE, hypothesis_id, SUPPORTED,
                   [_signal("distinct_codes", len(codes))])

    # --- C09 repeated-mention agreement --------------------------------------
    for edge in graph.edges_of(REL_SAME_SURFACE):
        subject = f"{edge.source_id}|{edge.target_id}"
        left = links_by_id.get(edge.source_id)
        right = links_by_id.get(edge.target_id)
        left_assertion = assertion_by_id.get(edge.source_id)
        right_assertion = assertion_by_id.get(edge.target_id)
        left_codes = frozenset(c.code for c in left.candidates) if left else frozenset()
        right_codes = frozenset(
            c.code for c in right.candidates) if right else frozenset()
        left_labels = frozenset(left_assertion.labels) if left_assertion else frozenset()
        right_labels = frozenset(
            right_assertion.labels) if right_assertion else frozenset()
        if not left_codes and not right_codes and left_labels == right_labels:
            record(RULE_REPEATED_MENTION, subject, SUPPORTED,
                   [_signal("assertions", "agree")],
                   hypothesis_ids=[edge.source_id, edge.target_id])
        elif left_codes and right_codes and not (left_codes & right_codes):
            # The same surface linked to disjoint code sets. Real disagreement.
            record(RULE_REPEATED_MENTION, subject, CONTRADICTED,
                   blocking=[_signal("disjoint_candidate_sets", "true")],
                   recommendation=REC_ESCALATE,
                   hypothesis_ids=[edge.source_id, edge.target_id],
                   candidate_codes=sorted(left_codes ^ right_codes)[:8])
        elif left_labels != right_labels:
            record(RULE_REPEATED_MENTION, subject, UNRESOLVED,
                   blocking=[_signal("left_labels", ",".join(sorted(left_labels))),
                             _signal("right_labels", ",".join(sorted(right_labels)))],
                   recommendation=REC_ESCALATE,
                   hypothesis_ids=[edge.source_id, edge.target_id])
        else:
            record(RULE_REPEATED_MENTION, subject, SUPPORTED,
                   [_signal("candidate_overlap", len(left_codes & right_codes))],
                   hypothesis_ids=[edge.source_id, edge.target_id])

    # --- C10 conflicting assertion evidence ----------------------------------
    for hypothesis_id, assertion in sorted(assertion_by_id.items()):
        if assertion.uncertain:
            record(RULE_ASSERTION_CONFLICT, hypothesis_id, UNRESOLVED,
                   blocking=[_signal("assertion_uncertain", "true"),
                             _signal("source", assertion.source)],
                   recommendation=REC_ESCALATE)
        elif len(assertion.labels) >= 3:
            # All three labels at once was the Audit-0052 defect. Held here too, so a
            # regression is reported rather than merely failing a unit test.
            record(RULE_ASSERTION_CONFLICT, hypothesis_id, CONTRADICTED,
                   blocking=[_signal("labels", ",".join(assertion.labels))],
                   fatal=True, recommendation=REC_WITHHOLD)
        else:
            record(RULE_ASSERTION_CONFLICT, hypothesis_id, SUPPORTED,
                   [_signal("labels", ",".join(assertion.labels) or "none")])

    # --- C11 unresolved structured medication conflict -----------------------
    for h in hypotheses:
        if h.entity_type != "THUỐC":
            record(RULE_MED_CONFLICT, h.hypothesis_id, NOT_APPLICABLE)
            continue
        med_result = links_by_id.get(h.hypothesis_id)
        suppressed = [w for w in (med_result.warnings if med_result else ())
                      if w.startswith("suppressed:")]
        retained = bool(med_result and med_result.candidates)
        if suppressed and not retained:
            # Every candidate conflicted with the mention's structured evidence.
            record(RULE_MED_CONFLICT, h.hypothesis_id, CONTRADICTED,
                   blocking=[_signal("suppressed", value) for value in suppressed[:6]],
                   recommendation=REC_ESCALATE)
        elif suppressed:
            record(RULE_MED_CONFLICT, h.hypothesis_id, UNRESOLVED,
                   [_signal("retained_candidates",
                            len(med_result.candidates) if med_result else 0)],
                   [_signal("suppressed", value) for value in suppressed[:6]],
                   recommendation=REC_EMIT_WITH_CAUTION)
        else:
            record(RULE_MED_CONFLICT, h.hypothesis_id, SUPPORTED,
                   [_signal("structured_conflicts", 0)])

    # --- C12 unsafe or unsupported treats ------------------------------------
    for medication_id, diagnosis_id in graph.declined_treats:
        record(RULE_UNSAFE_TREATS, f"{medication_id}|{diagnosis_id}", UNRESOLVED,
               blocking=[_signal("evidence", "co_occurrence_only")],
               recommendation=REC_WITHHOLD,
               hypothesis_ids=[medication_id, diagnosis_id])
    for edge in graph.edges_of(REL_TREATS):
        subject = f"{edge.source_id}|{edge.target_id}"
        record(RULE_UNSAFE_TREATS, subject, SUPPORTED,
               [_signal("evidence_source", edge.evidence_source),
                *(_signal("provenance", p) for p in edge.provenance[:2])],
               hypothesis_ids=[edge.source_id, edge.target_id],
               edge_keys=[f"{edge.relation}:{subject}"])

    # Candidate-count sanity: `has_candidate` edges must agree with the linker.
    for hypothesis_id, result in sorted(links_by_id.items()):
        edge_count = len(graph.neighbours(hypothesis_id, REL_HAS_CANDIDATE))
        if result.candidates and edge_count != len(result.candidates):
            issues.append(ConsistencyIssue(
                rule=RULE_CANDIDATE_TYPE, subject_id=hypothesis_id,
                verdict=CONTRADICTED, fatal=False,
                recommendation=REC_ESCALATE, hypothesis_ids=(hypothesis_id,),
                blocking=(_signal("linker_candidates", len(result.candidates)),
                          _signal("graph_edges", edge_count))))

    return GraphConsistencyReport(
        document_id=document_id,
        decisions=tuple(sorted(
            decisions, key=lambda d: (d.rule, d.subject_id, d.verdict))),
        issues=tuple(sorted(
            issues, key=lambda i: (not i.fatal, i.rule, i.subject_id))),
        graph_hash=graph.graph_hash,
    )


__all__ = [
    "ASSERTION_ELIGIBLE",
    "CONSISTENCY_VERSION",
    "CONTRADICTED",
    "HISTORY_SECTIONS",
    "NOT_APPLICABLE",
    "REC_EMIT",
    "REC_EMIT_WITH_CAUTION",
    "REC_ESCALATE",
    "REC_WITHHOLD",
    "RULES",
    "RULE_ASSERTION_CONFLICT",
    "RULE_ASSERTION_TYPE",
    "RULE_CANDIDATE_TYPE",
    "RULE_DUPLICATE_CANDIDATE",
    "RULE_ICD_HIERARCHY",
    "RULE_LAB_PAIR",
    "RULE_MED_CONFLICT",
    "RULE_OVERLAP",
    "RULE_REPEATED_MENTION",
    "RULE_RXNORM_STRUCTURED",
    "RULE_SECTION_COMPAT",
    "RULE_UNSAFE_TREATS",
    "SUPPORTED",
    "UNRESOLVED",
    "VERDICTS",
    "ConsistencyDecision",
    "ConsistencyIssue",
    "GraphConsistencyReport",
    "SupportSignal",
    "evaluate_consistency",
    "ontology_for",
]
