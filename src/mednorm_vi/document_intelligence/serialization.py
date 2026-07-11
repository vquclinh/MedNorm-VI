"""Deterministic debug serialization for the L1 DocumentGraph.

This is a debug/inspection artifact, NOT organizer prediction output. It fully
captures the graph (source metadata, normalized views + alignments, and every
structural node) so serialize → deserialize → serialize is stable, and a
determinism hash can be compared across runs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .alignment import CharAlignment
from .io import DocumentSource
from .models import (
    DocumentGraph,
    DocumentWarning,
    NodeKind,
    NormalizationStageRecord,
    NormalizedDocument,
    StructuralNode,
)

SCHEMA_VERSION = "l1-graph-1"

_NODE_FIELDS = (
    "node_id", "kind", "start", "end", "parent_id", "children", "category",
    "header_start", "header_end", "confidence", "matched_rule", "prior_label",
    "prior_strength", "marker_start", "marker_end", "content_start", "content_end",
    "indent", "depth", "row_kind", "is_punctuation", "whitespace_before",
    "token_category", "normalized_text", "line_id", "segment_id", "section_id",
    "provenance", "warnings", "attributes",
)


def _node_to_dict(node: StructuralNode) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in _NODE_FIELDS:
        value = getattr(node, f)
        if f == "kind":
            value = node.kind.value
        elif f in ("children", "warnings"):
            value = list(value)
        out[f] = value
    return out


def _node_from_dict(d: dict[str, Any]) -> StructuralNode:
    return StructuralNode(
        node_id=d["node_id"], kind=NodeKind(d["kind"]), start=d["start"], end=d["end"],
        parent_id=d.get("parent_id"), children=tuple(d.get("children", [])),
        category=d.get("category"), header_start=d.get("header_start"),
        header_end=d.get("header_end"), confidence=d.get("confidence"),
        matched_rule=d.get("matched_rule"), prior_label=d.get("prior_label"),
        prior_strength=d.get("prior_strength"), marker_start=d.get("marker_start"),
        marker_end=d.get("marker_end"), content_start=d.get("content_start"),
        content_end=d.get("content_end"), indent=d.get("indent"), depth=d.get("depth"),
        row_kind=d.get("row_kind"), is_punctuation=d.get("is_punctuation"),
        whitespace_before=d.get("whitespace_before"), token_category=d.get("token_category"),
        normalized_text=d.get("normalized_text"), line_id=d.get("line_id"),
        segment_id=d.get("segment_id"), section_id=d.get("section_id"),
        provenance=d.get("provenance"), warnings=tuple(d.get("warnings", [])),
        attributes=dict(d.get("attributes", {})),
    )


def _view_to_dict(view: NormalizedDocument) -> dict[str, Any]:
    return {
        "name": view.name,
        "text": view.text,
        "alignment": {
            "original_length": view.alignment.original_length,
            "normalized_length": view.alignment.normalized_length,
            "o2n": list(view.alignment.o2n),
        },
        "stages": [
            {"name": s.name, "version": s.version, "input_length": s.input_length,
             "output_length": s.output_length, "transformation_count": s.transformation_count,
             "config_hash": s.config_hash, "warnings": list(s.warnings)}
            for s in view.stages
        ],
    }


def _view_from_dict(d: dict[str, Any]) -> NormalizedDocument:
    a = d["alignment"]
    alignment = CharAlignment(
        original_length=a["original_length"], normalized_length=a["normalized_length"],
        o2n=tuple(a["o2n"]))
    stages = tuple(
        NormalizationStageRecord(
            name=s["name"], version=s["version"], input_length=s["input_length"],
            output_length=s["output_length"], transformation_count=s["transformation_count"],
            config_hash=s["config_hash"], warnings=tuple(s.get("warnings", [])))
        for s in d.get("stages", [])
    )
    return NormalizedDocument(name=d["name"], text=d["text"], alignment=alignment, stages=stages)


def _source_to_dict(s: DocumentSource) -> dict[str, Any]:
    return {
        "source_path": s.source_path, "encoding": s.encoding, "byte_length": s.byte_length,
        "char_length": s.char_length, "sha256": s.sha256, "newline_style": s.newline_style,
        "has_bom": s.has_bom, "bom_policy": s.bom_policy, "bom_stripped": s.bom_stripped,
    }


def to_debug_dict(graph: DocumentGraph) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": graph.document_id,
        "builder_version": graph.builder_version,
        "config_hash": graph.config_hash,
        "source": _source_to_dict(graph.source),
        "original_text": graph.original_text,
        "normalized": {name: _view_to_dict(graph.normalized[name])
                       for name in sorted(graph.normalized)},
        "nodes": [_node_to_dict(node) for node in graph.nodes],
        "warnings": [{"code": w.code, "message": w.message, "node_id": w.node_id}
                     for w in graph.warnings],
    }


def from_debug_dict(d: dict[str, Any]) -> DocumentGraph:
    src = d["source"]
    source = DocumentSource(
        source_path=src.get("source_path"), encoding=src["encoding"],
        byte_length=src["byte_length"], char_length=src["char_length"], sha256=src["sha256"],
        newline_style=src["newline_style"], has_bom=src["has_bom"], bom_policy=src["bom_policy"],
        bom_stripped=src.get("bom_stripped", False))
    return DocumentGraph(
        document_id=d["document_id"], source=source, original_text=d["original_text"],
        normalized={name: _view_from_dict(v) for name, v in d.get("normalized", {}).items()},
        nodes=tuple(_node_from_dict(nd) for nd in d.get("nodes", [])),
        warnings=tuple(DocumentWarning(w["code"], w["message"], w.get("node_id"))
                       for w in d.get("warnings", [])),
        builder_version=d["builder_version"], config_hash=d["config_hash"])


def to_json(graph: DocumentGraph) -> str:
    return json.dumps(to_debug_dict(graph), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def determinism_hash(graph: DocumentGraph) -> str:
    canonical = json.dumps(
        to_debug_dict(graph), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["to_debug_dict", "from_debug_dict", "to_json", "determinism_hash", "SCHEMA_VERSION"]
