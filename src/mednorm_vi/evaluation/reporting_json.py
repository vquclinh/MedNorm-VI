"""Deterministic JSON/JSONL report writers.

Writes: summary.json, per_document.json, entity_pairs.jsonl,
unmatched_entities.jsonl, diagnostics.jsonl, config_snapshot.json, and
replay_manifest.json. All are deterministic (no timestamps inside the data
reports — the timestamp lives only in the replay manifest and the HTML banner).
UTF-8, ``ensure_ascii=False``. Returns each file's sha256 for the replay manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import EvaluationOutcome
from .models import PerEntityScore, ReplayManifest, jsonable
from .replay import sha256_text

DATA_REPORT_FILES = (
    "summary.json",
    "per_document.json",
    "entity_pairs.jsonl",
    "unmatched_entities.jsonl",
    "diagnostics.jsonl",
    "config_snapshot.json",
)


def _dump_json(path: Path, obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows]
    text = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)


def _summary(outcome: EvaluationOutcome) -> dict[str, Any]:
    c = outcome.corpus
    cfg = outcome.config
    return {
        "banner": "PROVISIONAL LOCAL EVALUATOR",
        "evaluator_version": cfg.evaluator_version,
        "final_score": c.final_score,
        "text_score": c.text_score,
        "assertions_score": c.assertions_score,
        "candidates_score": c.candidates_score,
        "matching_strategy": cfg.matching_strategy,
        "tokenization": cfg.tokenization,
        "aggregation_policy": cfg.aggregation_policy,
        "clipping_enabled": cfg.clipping_enabled,
        "final_weights": {
            "text": cfg.weight_text,
            "assertions": cfg.weight_assertions,
            "candidates": cfg.weight_candidates,
        },
        "n_documents": c.n_documents,
        "n_entities": c.n_entities,
        "n_matched": c.n_matched,
        "n_missing": c.n_missing,
        "n_spurious": c.n_spurious,
        "score_by_type": c.per_type,
        "score_by_case": c.per_case,
        "diagnostic_counts": outcome.diagnostics.counts,
    }


def _per_document(outcome: EvaluationOutcome) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in outcome.per_document:
        rows.append({
            "document_id": d.document_id,
            "provenance": d.provenance.value if d.provenance else None,
            "final_score": d.final_score,
            "text_score": d.text_score,
            "assertions_score": d.assertions_score,
            "candidates_score": d.candidates_score,
            "n_gt": d.n_gt,
            "n_pred": d.n_pred,
            "n_matched": d.n_matched,
            "n_missing": d.n_missing,
            "n_spurious": d.n_spurious,
        })
    return rows


def _entity_pair_row(s: PerEntityScore) -> dict[str, Any]:
    assert s.pair is not None and s.wer is not None
    p, w = s.pair, s.wer
    row: dict[str, Any] = {
        "document_id": p.document_id,
        "gt_index": p.gt_index,
        "pred_index": p.pred_index,
        "entity_type": p.entity_type,
        "gt_text": p.gt_text,
        "pred_text": p.pred_text,
        "gt_position": list(p.gt_position),
        "pred_position": list(p.pred_position),
        "matching_strategy": p.strategy,
        "matching_cost": p.cost,
        "tokenization": w.tokenization,
        "wer_substitutions": w.substitutions,
        "wer_deletions": w.deletions,
        "wer_insertions": w.insertions,
        "wer_ref_tokens": w.ref_tokens,
        "raw_wer": w.raw_wer,
        "raw_text_score": w.raw_text_score,
        "clipped_text_score": w.clipped_text_score,
        "text_score": s.text_score,
        "route_case": s.route_case,
        "section": s.section,
        "diagnostics": list(s.diagnostics),
    }
    if s.assertions is not None:
        row["assertion_intersection"] = list(s.assertions.intersection)
        row["assertion_union"] = list(s.assertions.union)
        row["assertion_jaccard"] = s.assertions.jaccard
    if s.candidates is not None:
        row["candidate_intersection"] = list(s.candidates.intersection)
        row["candidate_union"] = list(s.candidates.union)
        row["candidate_jaccard"] = s.candidates.jaccard
        row["candidate_weight"] = s.candidates.weight
    return row


def _unmatched_row(s: PerEntityScore) -> dict[str, Any]:
    return {
        "document_id": s.document_id,
        "slot_kind": s.slot_kind,
        "entity_type": s.entity_type,
        "route_case": s.route_case,
        "section": s.section,
        "diagnostics": list(s.diagnostics),
    }


def write_data_reports(report_dir: str | Path, outcome: EvaluationOutcome,
                       config_mapping: dict[str, Any]) -> dict[str, str]:
    """Write the deterministic data reports; return {filename: sha256}."""
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    hashes["summary.json"] = _dump_json(root / "summary.json", _summary(outcome))
    hashes["per_document.json"] = _dump_json(root / "per_document.json", _per_document(outcome))

    pairs = [_entity_pair_row(s) for s in outcome.slots if s.slot_kind == "matched"]
    hashes["entity_pairs.jsonl"] = _dump_jsonl(root / "entity_pairs.jsonl", pairs)

    unmatched = [_unmatched_row(s) for s in outcome.slots if s.slot_kind != "matched"]
    hashes["unmatched_entities.jsonl"] = _dump_jsonl(root / "unmatched_entities.jsonl", unmatched)

    diags = [jsonable(d) for d in outcome.diagnostics.items]
    hashes["diagnostics.jsonl"] = _dump_jsonl(root / "diagnostics.jsonl", diags)

    snapshot = {"raw_config": config_mapping, "resolved_config": jsonable(outcome.config)}
    hashes["config_snapshot.json"] = _dump_json(root / "config_snapshot.json", snapshot)
    return hashes


def write_replay_manifest(report_dir: str | Path, manifest: ReplayManifest) -> str:
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    return _dump_json(root / "replay_manifest.json", jsonable(manifest))


__all__ = ["write_data_reports", "write_replay_manifest", "DATA_REPORT_FILES"]
