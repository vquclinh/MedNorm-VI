"""Replay + reproducibility helpers.

Records evaluator version, config hash, per-file and directory hashes, Python /
OS / Git state, and the resolved strategy names, so a run with identical inputs
and configuration reproduces identical scores and deterministic report data.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import EvaluationConfig, ReplayManifest


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON string (sorted keys, compact) for hashing."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_mapping(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def directory_file_hashes(directory: str | Path, suffix: str = ".json") -> dict[str, str]:
    """Map filename -> sha256 for every matching file, sorted by name."""
    root = Path(directory)
    if not root.is_dir():
        return {}
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix == suffix)
    return {p.name: sha256_file(p) for p in files}


def aggregate_dir_hash(file_hashes: dict[str, str]) -> str:
    """Order-independent hash of a directory's file-hash map."""
    return hash_mapping(dict(sorted(file_hashes.items())))


def git_state(cwd: str | Path | None = None) -> tuple[str | None, bool | None]:
    """Return (commit, dirty) or (None, None) if git is unavailable."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout
        return commit, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_replay_manifest(
    *,
    config: EvaluationConfig,
    config_mapping: dict[str, Any],
    ground_truth_dir: str | Path,
    predictions_dir: str | Path,
    ground_truth_provenance: str,
    prediction_experiment_id: str | None,
    timestamp: str,
) -> ReplayManifest:
    gt_hashes = directory_file_hashes(ground_truth_dir)
    pred_hashes = directory_file_hashes(predictions_dir)
    commit, dirty = git_state()
    return ReplayManifest(
        evaluator_version=config.evaluator_version,
        config_hash=hash_mapping(config_mapping),
        ground_truth_dir=str(ground_truth_dir),
        predictions_dir=str(predictions_dir),
        ground_truth_file_hashes=gt_hashes,
        prediction_file_hashes=pred_hashes,
        ground_truth_dir_hash=aggregate_dir_hash(gt_hashes),
        predictions_dir_hash=aggregate_dir_hash(pred_hashes),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        git_commit=commit,
        git_dirty=dirty,
        matching_strategy=config.matching_strategy,
        tokenization=config.tokenization,
        aggregation_policy=config.aggregation_policy,
        clipping_enabled=config.clipping_enabled,
        ground_truth_provenance=ground_truth_provenance,
        prediction_experiment_id=prediction_experiment_id,
        timestamp_utc=timestamp,
    )


__all__ = [
    "sha256_bytes",
    "sha256_text",
    "sha256_file",
    "canonical_json",
    "hash_mapping",
    "directory_file_hashes",
    "aggregate_dir_hash",
    "git_state",
    "utc_timestamp",
    "build_replay_manifest",
]
