"""Load the Phase 2A corpus-analysis cue inventory (config, not Python)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Compiled cue groups + descriptive thresholds. Purely for measurement."""

    analysis_version: str
    laboratory: dict[str, tuple[re.Pattern[str], ...]]
    medication: dict[str, tuple[re.Pattern[str], ...]]
    imaging: dict[str, tuple[re.Pattern[str], ...]]
    list_dominant_min_share: float
    narrative_dominant_max_share: float
    length_buckets: tuple[int, ...]
    config_hash: str = ""
    cue_group_names: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _compile(groups: dict[str, Any]) -> dict[str, tuple[re.Pattern[str], ...]]:
    out: dict[str, tuple[re.Pattern[str], ...]] = {}
    for name, patterns in (groups or {}).items():
        out[str(name)] = tuple(
            re.compile(str(p), re.IGNORECASE | re.UNICODE) for p in (patterns or [])
        )
    return out


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    import yaml  # type: ignore[import-untyped]

    doc: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    lab = _compile(doc.get("laboratory", {}))
    med = _compile(doc.get("medication", {}))
    img = _compile(doc.get("imaging", {}))
    profile = doc.get("profile", {}) or {}
    config_hash = hashlib.sha256(
        json.dumps(doc, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return AnalysisConfig(
        analysis_version=str(doc.get("analysis_version", "corpus-2a-1")),
        laboratory=lab, medication=med, imaging=img,
        list_dominant_min_share=float(profile.get("list_dominant_min_share", 0.5)),
        narrative_dominant_max_share=float(profile.get("narrative_dominant_max_share", 0.2)),
        length_buckets=tuple(int(b) for b in profile.get("length_buckets", [])),
        config_hash=config_hash,
        cue_group_names={"laboratory": tuple(sorted(lab)),
                         "medication": tuple(sorted(med)),
                         "imaging": tuple(sorted(img))})


__all__ = ["AnalysisConfig", "load_analysis_config"]
