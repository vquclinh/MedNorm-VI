"""Governed corpus split identity and resolution (spec §15.2).

Spec §15.2 requires that "every leaderboard decision records config, seed, Git
commit, and artifact hash", and spec §18 requires the local evaluator to be the
only judge of a change. Both fail the moment a run resolves its data by *name*: a
directory called ``validation`` in two runtimes is two different corpora, and a
metric computed against the wrong one is not comparable to anything.

So splits are resolved by **authoritative SHA-256**, never by name or path. A
resolution either finds a file whose bytes hash to the recorded digest or it
raises; there is no nearest match and no fallback.

``internal_test`` is refused outright. It is the held-out split every reported
number depends on, and it is not selectable through this module by any caller — a
split name is not a permission.

This module is expert-independent. It previously lived inside one expert's
contract file, which meant every other consumer imported that expert to identify
the shared corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .phase2.common import sha256_file

GOVERNED_SPLIT_CONTRACT_VERSION = "governed-split-v1"

# Authoritative digests of the governed training corpus v1
# (docs/training/governed_corpus_v1.md; verified in Audits 0020, 0032 and 0042).
GOVERNED_TRAIN_SHA256 = (
    "892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a")
GOVERNED_VALIDATION_SHA256 = (
    "ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103")

# Splits no training or ablation path may resolve. `internal_test` is held out.
FORBIDDEN_SPLIT_NAMES: frozenset[str] = frozenset({"internal_test"})

GOVERNED_SPLIT_SHA256: dict[str, str] = {
    "train": GOVERNED_TRAIN_SHA256,
    "validation": GOVERNED_VALIDATION_SHA256,
}


class GovernedSplitError(ValueError):
    """Raised when a split is forbidden or cannot be identified by digest."""


def assert_split_allowed(split: str) -> str:
    """Refuse a forbidden split by name, with the reason."""
    if split in FORBIDDEN_SPLIT_NAMES:
        raise GovernedSplitError(
            f"split {split!r} is held out and must never be resolved for "
            "training, threshold search or arm selection; it is the split every "
            "reported number depends on")
    return split


@dataclass(frozen=True, slots=True)
class GovernedSplitResolution:
    """One split, located by digest, with the path that actually matched."""

    split: str
    path: Path
    sha256: str


def resolve_governed_split_by_sha256(
    *, split: str, expected_sha256: str, search_roots: Sequence[str | Path],
) -> GovernedSplitResolution:
    """Locate a governed JSONL split by authoritative SHA-256, not by stale name."""
    assert_split_allowed(split)
    candidates: list[Path] = []
    for root in search_roots:
        path = Path(root)
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(sorted(path.rglob("*.jsonl")))
    for path in candidates:
        if path.is_file() and sha256_file(path) == expected_sha256:
            return GovernedSplitResolution(
                split=split, path=path, sha256=expected_sha256)
    raise FileNotFoundError(
        f"could not locate governed {split} split with SHA-256 {expected_sha256}; "
        f"searched {len(candidates)} candidate file(s) under "
        f"{[str(root) for root in search_roots]}")


def resolve_governed_splits(
    search_roots: Sequence[str | Path],
    *,
    splits: Sequence[str] = ("train", "validation"),
) -> dict[str, GovernedSplitResolution]:
    """Resolve several governed splits at once. Any miss raises."""
    resolved: dict[str, GovernedSplitResolution] = {}
    for split in splits:
        digest = GOVERNED_SPLIT_SHA256.get(split)
        if digest is None:
            raise GovernedSplitError(
                f"no authoritative digest is recorded for split {split!r}; "
                f"known splits are {sorted(GOVERNED_SPLIT_SHA256)}")
        resolved[split] = resolve_governed_split_by_sha256(
            split=split, expected_sha256=digest, search_roots=search_roots)
    return resolved


__all__ = [
    "FORBIDDEN_SPLIT_NAMES",
    "GOVERNED_SPLIT_CONTRACT_VERSION",
    "GOVERNED_SPLIT_SHA256",
    "GOVERNED_TRAIN_SHA256",
    "GOVERNED_VALIDATION_SHA256",
    "GovernedSplitError",
    "GovernedSplitResolution",
    "assert_split_allowed",
    "resolve_governed_split_by_sha256",
    "resolve_governed_splits",
]
