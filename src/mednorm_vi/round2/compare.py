"""Compare current task/data contracts with an upgraded Round-2 descriptor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    changed_keys: tuple[str, ...]
    reusable_checkpoints: tuple[str, ...] = field(default_factory=tuple)
    requires_retraining: tuple[str, ...] = field(default_factory=tuple)
    report_hash: str = ""


_TRACKED_KEYS = (
    "task_schema",
    "file_layout",
    "labels",
    "assertions",
    "candidate_contracts",
    "input_distribution",
    "headings_layouts",
    "position_policy",
    "kb_versions",
)


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"task descriptor must be a JSON object: {path}")
    return payload


def compare_task_descriptors(
    current_path: str | Path, upgraded_path: str | Path
) -> ComparisonReport:
    current = _load(current_path)
    upgraded = _load(upgraded_path)
    changed = tuple(key for key in _TRACKED_KEYS if current.get(key) != upgraded.get(key))
    retrain: list[str] = []
    if any(key in changed for key in ("labels", "assertions", "candidate_contracts")):
        retrain.append("resolver_and_decoders")
    if "input_distribution" in changed or "headings_layouts" in changed:
        retrain.append("l1_l2_calibration")
    if "kb_versions" in changed:
        retrain.append("linker_rerankers")
    reusable = tuple(str(v) for v in upgraded.get("reusable_checkpoints", []))
    payload = {"changed": changed, "reusable": reusable, "retrain": retrain}
    report_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ComparisonReport(
        changed_keys=changed,
        reusable_checkpoints=reusable,
        requires_retraining=tuple(sorted(retrain)),
        report_hash=report_hash,
    )


__all__ = ["ComparisonReport", "compare_task_descriptors"]
