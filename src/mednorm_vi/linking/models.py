"""Candidate-linking contracts for ICD-10 and RxNorm specialists."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LinkedCandidate:
    code: str
    score: float
    channels: tuple[str, ...] = field(default_factory=tuple)
    snapshot_id: str = ""
    evidence: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class LinkerResult:
    mention_id: str
    candidates: tuple[LinkedCandidate, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


__all__ = ["LinkedCandidate", "LinkerResult"]
