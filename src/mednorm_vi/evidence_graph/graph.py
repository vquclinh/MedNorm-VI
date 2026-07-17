"""Clinical Evidence Graph linking proposals, hypotheses, assertions, and candidates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..linking.models import LinkerResult
from ..resolution.models import EntityHypothesis
from ..specialists.assertion import AssertionDecision


@dataclass(frozen=True, slots=True)
class EvidenceNode:
    node_id: str
    node_type: str
    payload_hash: str


@dataclass(frozen=True, slots=True)
class EvidenceEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class ClinicalEvidenceGraph:
    document_id: str
    nodes: tuple[EvidenceNode, ...]
    edges: tuple[EvidenceEdge, ...]

    @property
    def graph_hash(self) -> str:
        payload = {
            "document_id": self.document_id,
            "nodes": [
                {"node_id": n.node_id, "node_type": n.node_type, "payload_hash": n.payload_hash}
                for n in self.nodes
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation,
                    "weight": e.weight,
                }
                for e in self.edges
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _payload_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, default=str, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_evidence_graph(
    document_id: str,
    hypotheses: tuple[EntityHypothesis, ...],
    assertions: tuple[AssertionDecision, ...],
    link_results: tuple[LinkerResult, ...] = (),
) -> ClinicalEvidenceGraph:
    nodes: list[EvidenceNode] = []
    edges: list[EvidenceEdge] = []
    for h in hypotheses:
        nodes.append(EvidenceNode(h.hypothesis_id, "hypothesis", _payload_hash(h)))
        for proposal_id in h.source_proposal_ids:
            pid = f"proposal:{proposal_id}"
            nodes.append(EvidenceNode(pid, "proposal", _payload_hash(proposal_id)))
            edges.append(EvidenceEdge(pid, h.hypothesis_id, "supports"))
    for decision in assertions:
        aid = f"assertion:{decision.hypothesis_id}"
        nodes.append(EvidenceNode(aid, "assertion", _payload_hash(decision)))
        edges.append(EvidenceEdge(decision.hypothesis_id, aid, "has_assertion"))
    for result in link_results:
        for candidate in result.candidates:
            cid = f"candidate:{result.mention_id}:{candidate.code}"
            nodes.append(EvidenceNode(cid, "candidate", _payload_hash(candidate)))
            edges.append(EvidenceEdge(result.mention_id, cid, "has_candidate", candidate.score))
    dedup_nodes = {node.node_id: node for node in nodes}
    return ClinicalEvidenceGraph(
        document_id=document_id,
        nodes=tuple(dedup_nodes[k] for k in sorted(dedup_nodes)),
        edges=tuple(sorted(edges, key=lambda e: (e.source_id, e.target_id, e.relation))),
    )


__all__ = ["ClinicalEvidenceGraph", "EvidenceEdge", "EvidenceNode", "build_evidence_graph"]
