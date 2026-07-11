"""Operations that mutate an experiment record: outputs, local + leaderboard scores.

Local scores (gold/silver/synthetic) are kept separate from the manually entered
leaderboard score. Nothing here contacts the network.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ..evaluation.replay import utc_timestamp
from .hashing import output_zip_hash, zip_prediction_hashes
from .models import ExperimentRecord, LeaderboardResult, LocalScore, LocalScoreKind
from .registry import ExperimentRegistry


def attach_output(
    registry: ExperimentRegistry, experiment_id: str, zip_path: str | Path
) -> ExperimentRecord:
    """Attach an output.zip: record its hash and per-file prediction hashes."""
    record = registry.load(experiment_id)
    updated = dataclasses.replace(
        record,
        output_dir=str(Path(zip_path).resolve().parent),
        output_zip_hash=output_zip_hash(zip_path),
        prediction_file_hashes=zip_prediction_hashes(zip_path),
    )
    registry.update(updated)
    return updated


def record_local_score(
    registry: ExperimentRegistry,
    experiment_id: str,
    *,
    kind: LocalScoreKind,
    final_score: float,
    text_score: float | None = None,
    assertions_score: float | None = None,
    candidates_score: float | None = None,
    dataset_id: str | None = None,
    evaluator_version: str | None = None,
) -> ExperimentRecord:
    """Record (or replace) a local provisional score of the given trust level."""
    record = registry.load(experiment_id)
    new_score = LocalScore(
        kind=kind,
        final_score=final_score,
        text_score=text_score,
        assertions_score=assertions_score,
        candidates_score=candidates_score,
        dataset_id=dataset_id,
        evaluator_version=evaluator_version,
        timestamp_utc=utc_timestamp(),
    )
    kept = tuple(s for s in record.local_scores if s.kind is not kind)
    updated = dataclasses.replace(record, local_scores=(*kept, new_score))
    registry.update(updated)
    return updated


def record_leaderboard(
    registry: ExperimentRegistry,
    experiment_id: str,
    *,
    score: float,
    submission_id: str | None = None,
    component_scores: dict[str, float] | None = None,
) -> ExperimentRecord:
    """Record a MANUALLY supplied leaderboard result (never scraped)."""
    record = registry.load(experiment_id)
    result = LeaderboardResult(
        score=score,
        submission_id=submission_id,
        timestamp_utc=utc_timestamp(),
        component_scores=dict(component_scores or {}),
    )
    updated = dataclasses.replace(record, leaderboard=result)
    registry.update(updated)
    return updated


def compare(records: list[ExperimentRecord]) -> list[dict[str, object]]:
    """Build a diffable comparison table across experiments (local vs leaderboard)."""
    rows: list[dict[str, object]] = []
    for r in records:
        gold = r.score_of_kind(LocalScoreKind.GOLD)
        silver = r.score_of_kind(LocalScoreKind.SILVER)
        synth = r.score_of_kind(LocalScoreKind.SYNTHETIC)
        rows.append({
            "experiment_id": r.experiment_id,
            "title": r.title,
            "git_commit": (r.git_commit or "")[:10],
            "git_dirty": r.git_dirty,
            "local_gold": gold.final_score if gold else None,
            "local_silver": silver.final_score if silver else None,
            "local_synthetic": synth.final_score if synth else None,
            "leaderboard": r.leaderboard.score if r.leaderboard else None,
            "output_zip_hash": (r.output_zip_hash or "")[:12] or None,
        })
    return rows


__all__ = ["attach_output", "record_local_score", "record_leaderboard", "compare"]
