"""Typed contracts for local leaderboard experiment tracking.

Local scores (gold / silver / synthetic) are kept strictly separate from the
manually entered leaderboard score. The tracker never scrapes the leaderboard or
calls any external service, and leaderboard score never replaces local error
analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LocalScoreKind(str, Enum):
    GOLD = "gold"
    SILVER = "silver"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True, slots=True)
class LocalScore:
    """A local provisional evaluation score, tagged by data trust level."""

    kind: LocalScoreKind
    final_score: float
    text_score: float | None = None
    assertions_score: float | None = None
    candidates_score: float | None = None
    dataset_id: str | None = None
    evaluator_version: str | None = None
    timestamp_utc: str | None = None


@dataclass(frozen=True, slots=True)
class LeaderboardResult:
    """A manually entered leaderboard result (never scraped)."""

    score: float
    submission_id: str | None = None
    timestamp_utc: str | None = None
    component_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """One tracked experiment. Serialized as a diff-friendly JSON file."""

    experiment_id: str
    title: str
    description: str = ""
    creation_time_utc: str = ""
    git_commit: str | None = None
    git_dirty: bool | None = None
    config_path: str | None = None
    config_hash: str | None = None
    model_profile: str | None = None
    model_versions: dict[str, str] = field(default_factory=dict)
    adapter_versions: dict[str, str] = field(default_factory=dict)
    data_versions: dict[str, str] = field(default_factory=dict)
    kb_versions: dict[str, str] = field(default_factory=dict)
    code_version: str | None = None
    seed: int | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    decoding_settings: dict[str, str] = field(default_factory=dict)
    output_dir: str | None = None
    output_zip_hash: str | None = None
    prediction_file_hashes: dict[str, str] = field(default_factory=dict)
    local_scores: tuple[LocalScore, ...] = ()
    leaderboard: LeaderboardResult | None = None
    change_summary: str = ""
    hypothesis: str = ""
    observed_result: str = ""
    conclusion: str = ""
    parent_experiment_id: str | None = None
    tags: tuple[str, ...] = ()
    notes: str = ""

    def score_of_kind(self, kind: LocalScoreKind) -> LocalScore | None:
        for s in self.local_scores:
            if s.kind is kind:
                return s
        return None


__all__ = ["LocalScoreKind", "LocalScore", "LeaderboardResult", "ExperimentRecord"]
