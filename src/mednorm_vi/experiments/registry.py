"""Local, diff-friendly experiment registry (JSON files, no network).

Deterministic experiment-id allocation (``EXP-0001`` ...). Records are written as
sorted-key JSON so Git diffs are clean. Accidental overwrite is refused unless
explicitly requested.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..evaluation.models import jsonable
from ..evaluation.replay import git_state, utc_timestamp
from .models import ExperimentRecord, LeaderboardResult, LocalScore, LocalScoreKind

_ID_RE = re.compile(r"^EXP-(\d+)$")


def _local_score_from_mapping(data: dict[str, Any]) -> LocalScore:
    return LocalScore(
        kind=LocalScoreKind(data["kind"]),
        final_score=float(data["final_score"]),
        text_score=data.get("text_score"),
        assertions_score=data.get("assertions_score"),
        candidates_score=data.get("candidates_score"),
        dataset_id=data.get("dataset_id"),
        evaluator_version=data.get("evaluator_version"),
        timestamp_utc=data.get("timestamp_utc"),
    )


def record_from_mapping(data: dict[str, Any]) -> ExperimentRecord:
    """Reconstruct an :class:`ExperimentRecord` from parsed JSON."""
    lb = data.get("leaderboard")
    leaderboard = None
    if lb:
        leaderboard = LeaderboardResult(
            score=float(lb["score"]),
            submission_id=lb.get("submission_id"),
            timestamp_utc=lb.get("timestamp_utc"),
            component_scores={
                str(k): float(v) for k, v in (lb.get("component_scores") or {}).items()
            },
        )
    return ExperimentRecord(
        experiment_id=data["experiment_id"],
        title=data.get("title", ""),
        description=data.get("description", ""),
        creation_time_utc=data.get("creation_time_utc", ""),
        git_commit=data.get("git_commit"),
        git_dirty=data.get("git_dirty"),
        config_path=data.get("config_path"),
        config_hash=data.get("config_hash"),
        model_profile=data.get("model_profile"),
        model_versions=dict(data.get("model_versions") or {}),
        adapter_versions=dict(data.get("adapter_versions") or {}),
        data_versions=dict(data.get("data_versions") or {}),
        kb_versions=dict(data.get("kb_versions") or {}),
        code_version=data.get("code_version"),
        seed=data.get("seed"),
        thresholds={str(k): float(v) for k, v in (data.get("thresholds") or {}).items()},
        decoding_settings=dict(data.get("decoding_settings") or {}),
        output_dir=data.get("output_dir"),
        output_zip_hash=data.get("output_zip_hash"),
        prediction_file_hashes=dict(data.get("prediction_file_hashes") or {}),
        local_scores=tuple(_local_score_from_mapping(s) for s in data.get("local_scores") or []),
        leaderboard=leaderboard,
        change_summary=data.get("change_summary", ""),
        hypothesis=data.get("hypothesis", ""),
        observed_result=data.get("observed_result", ""),
        conclusion=data.get("conclusion", ""),
        parent_experiment_id=data.get("parent_experiment_id"),
        tags=tuple(data.get("tags") or ()),
        notes=data.get("notes", ""),
    )


class ExperimentExistsError(RuntimeError):
    """Raised when saving would overwrite an existing experiment record."""


class ExperimentRegistry:
    """Filesystem-backed experiment registry."""

    def __init__(self, root: str | Path, *, id_format: str = "EXP-{:04d}") -> None:
        self.root = Path(root)
        self.id_format = id_format

    def _path(self, experiment_id: str) -> Path:
        return self.root / f"{experiment_id}.json"

    def list_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        ids = [p.stem for p in self.root.glob("EXP-*.json")]

        def _key(s: str) -> tuple[int, str]:
            m = _ID_RE.match(s)
            return (int(m.group(1)) if m else 1 << 60, s)

        return sorted(ids, key=_key)

    def exists(self, experiment_id: str) -> bool:
        return self._path(experiment_id).is_file()

    def next_id(self) -> str:
        max_n = 0
        for eid in self.list_ids():
            m = _ID_RE.match(eid)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return self.id_format.format(max_n + 1)

    def load(self, experiment_id: str) -> ExperimentRecord:
        path = self._path(experiment_id)
        if not path.is_file():
            raise FileNotFoundError(f"experiment {experiment_id} not found in {self.root}")
        return record_from_mapping(json.loads(path.read_text(encoding="utf-8")))

    def save(self, record: ExperimentRecord, *, overwrite: bool = False) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(record.experiment_id)
        if path.exists() and not overwrite:
            raise ExperimentExistsError(
                f"experiment {record.experiment_id} already exists; refusing to overwrite")
        text = json.dumps(jsonable(record), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        path.write_text(text, encoding="utf-8")
        return path

    def create(self, *, title: str, **kwargs: Any) -> ExperimentRecord:
        """Allocate the next id, capture git state + timestamp, and persist."""
        commit, dirty = git_state()
        record = ExperimentRecord(
            experiment_id=self.next_id(),
            title=title,
            creation_time_utc=utc_timestamp(),
            git_commit=commit,
            git_dirty=dirty,
            **kwargs,
        )
        self.save(record, overwrite=False)
        return record

    def update(self, record: ExperimentRecord) -> Path:
        """Persist an updated record (overwrite allowed for an existing id)."""
        if not self.exists(record.experiment_id):
            raise FileNotFoundError(f"cannot update missing experiment {record.experiment_id}")
        return self.save(record, overwrite=True)


__all__ = [
    "ExperimentRegistry",
    "ExperimentExistsError",
    "record_from_mapping",
]
