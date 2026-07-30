"""Clinical Evidence Graph (L6, spec §11).

Before Audit 0054 this graph carried three relations — ``supports``,
``has_assertion``, ``has_candidate`` — and nothing read it. Audit 0053 measured exactly
that: 333 ``has_assertion``, 4,000 ``has_candidate``, 333 ``supports`` over 200
documents, and no downstream consumer. It recorded the graph as *write-only*.

This module now builds every edge type spec §11 names **that current evidence actually
supports**, and each edge carries its own provenance so a reader can tell an edge
derived from E2's pairing from one derived from a character-interval comparison:

```text
supports      proposal -> hypothesis        L3/L4 provenance
has_assertion hypothesis -> assertion       L5 assertion decision
has_candidate hypothesis -> candidate       L5 linker
has_result    TEST_NAME -> TEST_RESULT      E2 pair_group_id preserved through L4
modified_by   hypothesis -> component       E1 ComponentSpan (structured only)
in_section    hypothesis -> section         L1 section node containing the span
overlaps      hypothesis <-> hypothesis     verified character intervals
same_surface  hypothesis <-> hypothesis     normalized surface equality
treats        MEDICATION -> DIAGNOSIS       explicit governed evidence ONLY
```

**``treats`` is the one edge that must be hard to create**, and it is. Spec §11 lists
it, but a medication and a diagnosis appearing in the same document says nothing about
treatment — a discharge summary mentions a dozen of each. So ``treats`` requires an
explicit textual trigger *between* the two mentions inside one sentence-scale window,
or a governed KB relation. Co-occurrence is never enough, and when only co-occurrence
exists the pair produces **no edge at all** rather than a weak one. The declined pair
is recorded, because "we considered this and refused to assert it" is information the
consistency layer reports.

Determinism: nodes are keyed and sorted by id, edges de-duplicated by
(source, target, relation) and sorted, and ``graph_hash`` covers every field including
provenance. The same input produces byte-identical serialization.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..document_intelligence.models import DocumentGraph, NodeKind
from ..linking.models import LinkerResult
from ..mention_factory.models import RelationProposal
from ..resolution.models import EntityHypothesis
from ..specialists.assertion import AssertionDecision

EVIDENCE_GRAPH_VERSION = "clinical-evidence-graph-v2"

# Relation vocabulary (spec §11).
REL_SUPPORTS = "supports"
REL_HAS_ASSERTION = "has_assertion"
REL_HAS_CANDIDATE = "has_candidate"
REL_HAS_RESULT = "has_result"
REL_MODIFIED_BY = "modified_by"
REL_IN_SECTION = "in_section"
REL_OVERLAPS = "overlaps"
REL_SAME_SURFACE = "same_surface"
REL_TREATS = "treats"

RELATIONS: tuple[str, ...] = (
    REL_SUPPORTS, REL_HAS_ASSERTION, REL_HAS_CANDIDATE, REL_HAS_RESULT,
    REL_MODIFIED_BY, REL_IN_SECTION, REL_OVERLAPS, REL_SAME_SURFACE, REL_TREATS,
)

# Evidence sources, so an edge's origin is never guessed from its relation.
EV_L3_PROVENANCE = "l3_proposal_provenance"
EV_L4_RESOLUTION = "l4_resolution"
EV_L5_ASSERTION = "l5_assertion_decision"
EV_L5_LINKER = "l5_linker"
EV_E1_COMPONENT = "e1_component_span"
EV_E2_PAIR_GROUP = "e2_pair_group"
EV_L1_SECTION = "l1_section_node"
EV_CHAR_INTERVAL = "verified_character_interval"
EV_NORMALIZED_SURFACE = "normalized_surface"
EV_EXPLICIT_TEXT = "explicit_text_trigger"
EV_GOVERNED_KB = "governed_kb_relation"

# Confidence tiers. Evidence strength, not probability.
TIER_DERIVED = "derived"      # mechanically implied by an upstream contract
TIER_EXPLICIT = "explicit"    # stated in the document or the locked KB
TIER_INFERRED = "inferred"    # computed by comparison (intervals, surfaces)

# Explicit Vietnamese treatment triggers. Deliberately short: a long permissive list
# would turn `treats` back into co-occurrence with extra steps.
_TREATS_TRIGGERS: tuple[str, ...] = (
    "dieu tri", "de dieu tri", "chi dinh cho", "chi dinh dieu tri", "dung cho",
    "dung de", "khang sinh cho", "kiem soat",
)
# Maximum characters between the two mentions for a trigger to bind them. One
# sentence-scale window; wider windows are how spurious clinical claims appear.
TREATS_WINDOW = 120


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(unicodedata.normalize("NFKC", stripped).casefold().split())


@dataclass(frozen=True, slots=True)
class EvidenceNode:
    node_id: str
    node_type: str
    payload_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id, "node_type": self.node_type,
            "payload_hash": self.payload_hash,
        }


@dataclass(frozen=True, slots=True)
class EvidenceEdge:
    """One typed edge with the provenance that justifies it."""

    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    evidence_source: str = EV_L4_RESOLUTION
    confidence_tier: str = TIER_DERIVED
    provenance: tuple[str, ...] = field(default_factory=tuple)
    rule: str = EVIDENCE_GRAPH_VERSION

    @property
    def key(self) -> tuple[str, str, str]:
        """Identity for de-duplication: one edge per (source, target, relation)."""
        return (self.source_id, self.target_id, self.relation)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id, "target_id": self.target_id,
            "relation": self.relation, "weight": self.weight,
            "evidence_source": self.evidence_source,
            "confidence_tier": self.confidence_tier,
            "provenance": list(self.provenance), "rule": self.rule,
        }


@dataclass(frozen=True, slots=True)
class ClinicalEvidenceGraph:
    document_id: str
    nodes: tuple[EvidenceNode, ...]
    edges: tuple[EvidenceEdge, ...]
    version: str = EVIDENCE_GRAPH_VERSION
    # `treats` pairs that had co-occurrence but no explicit evidence. Recorded so the
    # consistency layer can report a declined claim rather than silently dropping it.
    declined_treats: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def edges_of(self, relation: str) -> tuple[EvidenceEdge, ...]:
        return tuple(e for e in self.edges if e.relation == relation)

    def edge_counts(self) -> dict[str, int]:
        counts = {relation: 0 for relation in RELATIONS}
        for edge in self.edges:
            counts[edge.relation] = counts.get(edge.relation, 0) + 1
        return counts

    def neighbours(self, node_id: str, relation: str) -> tuple[str, ...]:
        return tuple(
            e.target_id for e in self.edges
            if e.source_id == node_id and e.relation == relation)

    @property
    def graph_hash(self) -> str:
        payload = {
            "version": self.version,
            "document_id": self.document_id,
            "nodes": [n.as_dict() for n in self.nodes],
            "edges": [e.as_dict() for e in self.edges],
            "declined_treats": [list(pair) for pair in self.declined_treats],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Deterministic summary. Carries no clinical text."""
        return {
            "evidence_graph_version": self.version,
            "document_id": self.document_id,
            "graph_hash": self.graph_hash,
            "node_count": len(self.nodes),
            "edge_counts": self.edge_counts(),
            "declined_treats": len(self.declined_treats),
            "contains_clinical_text": False,
        }


def _payload_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, default=str, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _section_for(document: DocumentGraph | None, start: int, end: int) -> str | None:
    """The narrowest L1 section node containing ``[start, end)``."""
    if document is None:
        return None
    best: tuple[int, str] | None = None
    for node in document.nodes:
        # `NodeKind` is a `str` Enum, so `str(node.kind)` renders "NodeKind.SECTION".
        # Compare the value, which equals "section" because the enum subclasses str.
        if getattr(node, "kind", None) != NodeKind.SECTION:
            continue
        if node.start <= start and end <= node.end:
            width = node.end - node.start
            if best is None or width < best[0]:
                best = (width, node.node_id)
    return best[1] if best else None


def _explicit_treats_evidence(
    original_text: str, medication: EntityHypothesis, diagnosis: EntityHypothesis
) -> str | None:
    """The trigger phrase binding a medication to a diagnosis, or ``None``.

    Requires the trigger to sit *between* the two mentions within one sentence-scale
    window. Co-occurrence alone returns ``None`` — which is the point.
    """
    first, second = sorted((medication, diagnosis), key=lambda h: h.start)
    gap_start, gap_end = first.end, second.start
    if gap_end <= gap_start or (gap_end - gap_start) > TREATS_WINDOW:
        return None
    gap = original_text[gap_start:gap_end]
    # A sentence terminator between the mentions breaks the claim.
    if any(ch in gap for ch in ".;\n"):
        return None
    normalized = _normalize(gap)
    return next((t for t in _TREATS_TRIGGERS if t in normalized), None)


def build_evidence_graph(
    document_id: str,
    hypotheses: tuple[EntityHypothesis, ...],
    assertions: tuple[AssertionDecision, ...],
    link_results: tuple[LinkerResult, ...] = (),
    *,
    document: DocumentGraph | None = None,
    relations: Sequence[RelationProposal] = (),
    kb_treats: Mapping[str, frozenset[str]] | None = None,
) -> ClinicalEvidenceGraph:
    """Build the L6 graph. Optional inputs unlock the edges that depend on them.

    ``document`` unlocks ``in_section`` and text-triggered ``treats``; ``relations``
    marks primary ``has_result`` pairings; ``kb_treats`` unlocks governed ``treats``.
    Omitting one produces fewer edges, never weaker ones — an edge type with no
    evidence is absent, not guessed.
    """
    nodes: list[EvidenceNode] = []
    edges: list[EvidenceEdge] = []
    original_text = document.original_text if document is not None else ""

    # --- supports / modified_by / in_section ---------------------------------
    for h in hypotheses:
        nodes.append(EvidenceNode(h.hypothesis_id, "hypothesis", _payload_hash(h)))
        for proposal_id in h.source_proposal_ids:
            pid = f"proposal:{proposal_id}"
            nodes.append(EvidenceNode(pid, "proposal", _payload_hash(proposal_id)))
            edges.append(EvidenceEdge(
                pid, h.hypothesis_id, REL_SUPPORTS,
                evidence_source=EV_L3_PROVENANCE, confidence_tier=TIER_DERIVED,
                provenance=(f"proposal:{proposal_id}",)))

        # modified_by: only from explicit structured components. A modifier the
        # grammar did not emit is not inferred from adjacency.
        for role, components in h.components_by_role().items():
            if role in {"name", "test_name"}:
                continue  # the head, not a modifier
            for component in components:
                start = int(component.get("start", 0))
                end = int(component.get("end", 0))
                cid = f"component:{h.hypothesis_id}:{role}:{start}:{end}"
                nodes.append(EvidenceNode(cid, "component", _payload_hash(component)))
                edges.append(EvidenceEdge(
                    h.hypothesis_id, cid, REL_MODIFIED_BY,
                    evidence_source=EV_E1_COMPONENT, confidence_tier=TIER_EXPLICIT,
                    provenance=(f"role:{role}", f"span:{start}:{end}")))

        section_id = _section_for(document, h.start, h.end)
        if section_id is not None:
            sid = f"section:{section_id}"
            nodes.append(EvidenceNode(sid, "section", _payload_hash(section_id)))
            edges.append(EvidenceEdge(
                h.hypothesis_id, sid, REL_IN_SECTION,
                evidence_source=EV_L1_SECTION, confidence_tier=TIER_EXPLICIT,
                provenance=(f"section_node:{section_id}",)))

    # --- has_assertion -------------------------------------------------------
    for decision in assertions:
        aid = f"assertion:{decision.hypothesis_id}"
        nodes.append(EvidenceNode(aid, "assertion", _payload_hash(decision)))
        edges.append(EvidenceEdge(
            decision.hypothesis_id, aid, REL_HAS_ASSERTION,
            evidence_source=EV_L5_ASSERTION, confidence_tier=TIER_DERIVED,
            provenance=(f"source:{decision.source}",
                        f"uncertain:{decision.uncertain}")))

    # --- has_candidate -------------------------------------------------------
    for result in link_results:
        for candidate in result.candidates:
            cid = f"candidate:{result.mention_id}:{candidate.code}"
            nodes.append(EvidenceNode(cid, "candidate", _payload_hash(candidate)))
            edges.append(EvidenceEdge(
                result.mention_id, cid, REL_HAS_CANDIDATE, candidate.score,
                evidence_source=EV_L5_LINKER, confidence_tier=TIER_DERIVED,
                provenance=(
                    f"snapshot:{candidate.snapshot_id}", *candidate.evidence[:4])))

    # --- has_result, from E2's preserved pair groups --------------------------
    # Audit 0052 preserved `has_result_pair_group_ids` to L4 for exactly this. The
    # pairing is E2's, not re-derived here from proximity.
    group_members: dict[str, list[EntityHypothesis]] = {}
    for h in hypotheses:
        for group_id in h.has_result_pair_group_ids:
            group_members.setdefault(group_id, []).append(h)
    primary_groups = {
        r.pair_group_id for r in relations
        if r.relation_type.upper() == "HAS_RESULT" and r.is_primary}
    for group_id, members in sorted(group_members.items()):
        names = sorted(
            (h for h in members if h.entity_type == "TÊN_XÉT_NGHIỆM"),
            key=lambda h: h.start)
        results = sorted(
            (h for h in members if h.entity_type == "KẾT_QUẢ_XÉT_NGHIỆM"),
            key=lambda h: h.start)
        for name in names:
            for result_h in results:
                edges.append(EvidenceEdge(
                    name.hypothesis_id, result_h.hypothesis_id, REL_HAS_RESULT,
                    evidence_source=EV_E2_PAIR_GROUP, confidence_tier=TIER_EXPLICIT,
                    provenance=(
                        f"pair_group:{group_id}",
                        f"primary:{group_id in primary_groups}")))

    # --- overlaps / same_surface, from verified coordinates -------------------
    ordered = sorted(hypotheses, key=lambda h: (h.start, h.end, h.hypothesis_id))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if right.start >= left.end:
                break  # sorted by start: nothing further can overlap `left`
            edges.append(EvidenceEdge(
                left.hypothesis_id, right.hypothesis_id, REL_OVERLAPS,
                evidence_source=EV_CHAR_INTERVAL, confidence_tier=TIER_INFERRED,
                provenance=(
                    f"left:{left.start}:{left.end}",
                    f"right:{right.start}:{right.end}",
                    f"same_type:{left.entity_type == right.entity_type}")))

    by_surface: dict[tuple[str, str], list[EntityHypothesis]] = {}
    for h in hypotheses:
        by_surface.setdefault((_normalize(h.text), h.entity_type), []).append(h)
    for (surface, _entity_type), members in sorted(by_surface.items()):
        if len(members) < 2 or not surface:
            continue
        chain = sorted(members, key=lambda h: (h.start, h.hypothesis_id))
        for left, right in zip(chain, chain[1:], strict=False):
            # Separate occurrence coordinates are preserved on both endpoints; the
            # edge links them without merging them.
            edges.append(EvidenceEdge(
                left.hypothesis_id, right.hypothesis_id, REL_SAME_SURFACE,
                evidence_source=EV_NORMALIZED_SURFACE, confidence_tier=TIER_INFERRED,
                provenance=(
                    f"left:{left.start}:{left.end}",
                    f"right:{right.start}:{right.end}")))

    # --- treats: explicit evidence only --------------------------------------
    declined: list[tuple[str, str]] = []
    medications = sorted(
        (h for h in hypotheses if h.entity_type == "THUỐC"), key=lambda h: h.start)
    diagnoses = sorted(
        (h for h in hypotheses if h.entity_type == "CHẨN_ĐOÁN"), key=lambda h: h.start)
    kb = kb_treats or {}
    for medication in medications:
        allowed = kb.get(medication.hypothesis_id, frozenset())
        for diagnosis in diagnoses:
            trigger = (_explicit_treats_evidence(original_text, medication, diagnosis)
                       if original_text else None)
            if trigger is not None:
                edges.append(EvidenceEdge(
                    medication.hypothesis_id, diagnosis.hypothesis_id, REL_TREATS,
                    evidence_source=EV_EXPLICIT_TEXT, confidence_tier=TIER_EXPLICIT,
                    provenance=(f"trigger:{trigger}",)))
            elif diagnosis.hypothesis_id in allowed:
                edges.append(EvidenceEdge(
                    medication.hypothesis_id, diagnosis.hypothesis_id, REL_TREATS,
                    evidence_source=EV_GOVERNED_KB, confidence_tier=TIER_EXPLICIT,
                    provenance=("governed_kb_relation",)))
            else:
                # Co-occurrence only. No edge — and the declined pair is recorded.
                declined.append((medication.hypothesis_id, diagnosis.hypothesis_id))

    dedup_nodes = {node.node_id: node for node in nodes}
    dedup_edges: dict[tuple[str, str, str], EvidenceEdge] = {}
    for edge in edges:
        # An edge into a node that does not exist is a defect, not data.
        if edge.source_id in dedup_nodes and edge.target_id in dedup_nodes:
            dedup_edges.setdefault(edge.key, edge)
    return ClinicalEvidenceGraph(
        document_id=document_id,
        nodes=tuple(dedup_nodes[key] for key in sorted(dedup_nodes)),
        edges=tuple(sorted(
            dedup_edges.values(),
            key=lambda e: (e.relation, e.source_id, e.target_id))),
        declined_treats=tuple(sorted(set(declined))),
    )


__all__ = [
    "EVIDENCE_GRAPH_VERSION",
    "EV_CHAR_INTERVAL",
    "EV_E1_COMPONENT",
    "EV_E2_PAIR_GROUP",
    "EV_EXPLICIT_TEXT",
    "EV_GOVERNED_KB",
    "EV_L1_SECTION",
    "EV_L3_PROVENANCE",
    "EV_L4_RESOLUTION",
    "EV_L5_ASSERTION",
    "EV_L5_LINKER",
    "EV_NORMALIZED_SURFACE",
    "RELATIONS",
    "REL_HAS_ASSERTION",
    "REL_HAS_CANDIDATE",
    "REL_HAS_RESULT",
    "REL_IN_SECTION",
    "REL_MODIFIED_BY",
    "REL_OVERLAPS",
    "REL_SAME_SURFACE",
    "REL_SUPPORTS",
    "REL_TREATS",
    "TIER_DERIVED",
    "TIER_EXPLICIT",
    "TIER_INFERRED",
    "TREATS_WINDOW",
    "ClinicalEvidenceGraph",
    "EvidenceEdge",
    "EvidenceNode",
    "build_evidence_graph",
]
