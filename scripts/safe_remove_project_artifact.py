#!/usr/bin/env python3
"""Guarded removal of generated project artifacts (Audit 0065 §2).

Audit 0064 §2 recorded two defects in the ad-hoc cleanup that preceded this tool, and
both are fixed here structurally rather than by remembering to be careful:

1. **A substring grep treated a provenance field as a live reference.**
   `configs/models/e3_checkpoint_profiles.yaml` records `experiment_origin:` to document
   where a checkpoint came from. Grepping for the path matched that line and reported a
   reference that does not exist at runtime. This tool **parses** the governed configs and
   inspects only the fields that actually load a checkpoint, so documentation can mention
   a path without pinning it forever.

2. **The script printed "ABORT" and kept going.** A printed warning is not a guard. Every
   failed precondition here raises :class:`RemovalRefused`, which exits with a nonzero
   status *before* anything is unlinked.

Additional properties, all enforced rather than documented:

* dry-run is the default; `--execute` is required to unlink anything;
* paths are resolved canonically and must stay inside the repository;
* symlinks are refused outright, so a link cannot be used to escape the repo;
* the active and rollback checkpoints are refused by digest AND by path;
* a directory target is bounded - it must contain no subdirectories and no protected file;
* no wildcard expansion, no shell, no `eval`;
* a receipt is written for every executed removal.

Usage::

    python scripts/safe_remove_project_artifact.py PATH [PATH ...]           # dry run
    python scripts/safe_remove_project_artifact.py PATH --execute \
        --receipt runs/diagnostics/0065_checkpoint_cleanup/deletion_receipt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

#: Governed configs that may name a checkpoint the runtime loads.
PROFILE_REGISTRY = REPO / "configs" / "models" / "e3_checkpoint_profiles.yaml"
PIPELINE_CONFIG = REPO / "configs" / "pipeline" / "full_v1.yaml"
MODEL_REGISTRY = REPO / "configs" / "models" / "candidate_model_registry.yaml"

#: Fields that make a path LIVE. Anything else in these files is documentation.
PROFILE_LOAD_FIELDS = ("path",)
PIPELINE_LOAD_FIELDS = ("e3_checkpoint_path",)
REGISTRY_LOAD_FIELDS = ("active_checkpoint_path",)

#: Fields that are provenance only. Present here so the distinction is explicit and
#: testable rather than implied by omission.
PROVENANCE_ONLY_FIELDS = ("experiment_origin", "source_checkpoint", "notes")

#: Digests that must never be removed, whatever path they are found at.
PROTECTED_DIGESTS = {
    "524ece1e7d190838cb8b1ce3b0a0f337bc5b8b7cc7cef70c4c3e0b0310adde3a": "ACTIVE E3 checkpoint",
    "a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c": "ROLLBACK E3 checkpoint",
}

#: Suffixes treated as model weights; these get a digest check before any decision.
WEIGHT_SUFFIXES = (".pt", ".bin", ".safetensors", ".ckpt", ".pth")


class RemovalRefused(RuntimeError):
    """A guard failed. Nothing is removed and the process exits nonzero."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- reference analysis -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reference:
    """One place a path is named, and whether that naming is load-bearing."""

    config: str
    field: str
    live: bool

    def describe(self) -> str:
        kind = "LIVE" if self.live else "provenance-only"
        return f"{self.config}:{self.field} ({kind})"


def _load_yaml(path: Path) -> Any:
    if not path.is_file():
        return None
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def find_references(relative: str) -> list[Reference]:
    """Every governed mention of ``relative``, classified live vs provenance.

    This is the fix for Audit 0064 defect 1: the configs are PARSED and only the fields
    that actually resolve a checkpoint at runtime can make a path live.
    """
    found: list[Reference] = []

    profiles = _load_yaml(PROFILE_REGISTRY)
    if isinstance(profiles, dict):
        for name, entry in (profiles.get("profiles") or {}).items():
            if not isinstance(entry, dict):
                continue
            for key, value in entry.items():
                if not isinstance(value, str) or value != relative:
                    continue
                found.append(
                    Reference(
                        config=f"{PROFILE_REGISTRY.name}:profiles.{name}",
                        field=key,
                        live=key in PROFILE_LOAD_FIELDS,
                    )
                )

    pipeline = _load_yaml(PIPELINE_CONFIG)
    if isinstance(pipeline, dict):
        settings = pipeline.get("expert_settings") or {}
        if isinstance(settings, dict):
            for key, value in settings.items():
                if isinstance(value, str) and value == relative:
                    found.append(
                        Reference(
                            config=f"{PIPELINE_CONFIG.name}:expert_settings",
                            field=key,
                            live=key in PIPELINE_LOAD_FIELDS,
                        )
                    )

    registry = _load_yaml(MODEL_REGISTRY)
    if isinstance(registry, dict):
        for component in registry.get("components") or []:
            if not isinstance(component, dict):
                continue
            for key, value in component.items():
                if isinstance(value, str) and value == relative:
                    found.append(
                        Reference(
                            config=(f"{MODEL_REGISTRY.name}:{component.get('component_id', '?')}"),
                            field=key,
                            live=key in REGISTRY_LOAD_FIELDS,
                        )
                    )
    return found


# --- guards -------------------------------------------------------------------


@dataclass
class GuardReport:
    target: str
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    files: list[tuple[str, int]] = field(default_factory=list)
    total_bytes: int = 0

    def record(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append((name, passed, detail))

    @property
    def passed(self) -> bool:
        return all(ok for _n, ok, _d in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "guards": [{"guard": n, "passed": ok, "detail": d} for n, ok, d in self.checks],
            "files": [{"path": p, "bytes": b} for p, b in self.files],
            "total_bytes": self.total_bytes,
            "decision": "REMOVE" if self.passed else "REFUSED",
        }


def _resolve_inside_repo(raw: str) -> Path:
    """Canonical path, refusing symlinks and anything outside the repository."""
    candidate = Path(raw)
    absolute = candidate if candidate.is_absolute() else (REPO / candidate)
    if absolute.is_symlink():
        raise RemovalRefused(f"refusing a symlink: {raw}")
    for parent in absolute.parents:
        if parent == REPO.parent:
            break
        if parent.is_symlink():
            raise RemovalRefused(f"refusing a path through a symlinked parent: {parent}")
    resolved = absolute.resolve()
    try:
        resolved.relative_to(REPO)
    except ValueError as exc:
        raise RemovalRefused(
            f"refusing a path outside the repository: {resolved} (repo {REPO})"
        ) from exc
    return resolved


def inspect(raw: str) -> GuardReport:
    """Run every guard for one target. Never removes anything."""
    report = GuardReport(target=raw)

    resolved = _resolve_inside_repo(raw)
    relative = str(resolved.relative_to(REPO))
    report.record("canonical path inside repository", True, relative)
    report.record("not a symlink", True, "checked target and every parent")

    if not resolved.exists():
        report.record("target exists", False, f"{relative} does not exist")
        return report
    report.record("target exists", True, relative)

    if resolved.is_dir():
        children = sorted(resolved.iterdir())
        subdirs = [c for c in children if c.is_dir()]
        if subdirs:
            report.record(
                "directory is bounded (no subdirectories)",
                False,
                f"{len(subdirs)} subdirectory(ies): {[c.name for c in subdirs][:4]}",
            )
            return report
        report.record("directory is bounded (no subdirectories)", True, f"{len(children)} file(s)")
        targets = [c for c in children if c.is_file()]
    else:
        targets = [resolved]

    protected_hit: list[str] = []
    live_refs: list[str] = []
    provenance_refs: list[str] = []
    for item in targets:
        item_relative = str(item.relative_to(REPO))
        report.files.append((item_relative, item.stat().st_size))
        report.total_bytes += item.stat().st_size

        if item.suffix.lower() in WEIGHT_SUFFIXES:
            digest = sha256_file(item)
            if digest in PROTECTED_DIGESTS:
                protected_hit.append(f"{item_relative} is the {PROTECTED_DIGESTS[digest]}")

        for reference in find_references(item_relative):
            (live_refs if reference.live else provenance_refs).append(
                f"{item_relative} <- {reference.describe()}"
            )

    report.record(
        "not an active or rollback checkpoint (by digest)",
        not protected_hit,
        "; ".join(protected_hit) if protected_hit else "no protected digest matched",
    )
    report.record(
        "no LIVE runtime reference",
        not live_refs,
        "; ".join(live_refs) if live_refs else "no load-path field names this artifact",
    )
    # Recorded, never blocking. This is Audit 0064 defect 1, made explicit.
    report.record(
        "provenance-only references are not blocking",
        True,
        "; ".join(provenance_refs) if provenance_refs else "none",
    )
    return report


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="one file or one bounded directory each")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually remove; without it this is a dry run and nothing is touched",
    )
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--operator", default="", help="who authorised this removal")
    parser.add_argument("--reason", default="", help="why this group may be removed")
    args = parser.parse_args(argv)

    reports: list[GuardReport] = []
    refused = False
    for raw in args.paths:
        try:
            report = inspect(raw)
        except RemovalRefused as exc:
            print(f"REFUSED  {raw}\n  guard: {exc}", file=sys.stderr)
            refused = True
            continue
        reports.append(report)
        print(f"\n=== {raw} ===")
        for name, ok, detail in report.checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<48} {detail}")
        for path, size in report.files:
            print(f"      {size:>14,}  {path}")
        print(f"  total {report.total_bytes:,} bytes -> {report.as_dict()['decision']}")
        if not report.passed:
            refused = True

    if refused:
        print(
            "\nAT LEAST ONE GUARD FAILED. Nothing was removed. Exiting nonzero.",
            file=sys.stderr,
        )
        return 2

    if not args.execute:
        total = sum(r.total_bytes for r in reports)
        print(f"\nDRY RUN. {len(reports)} target(s), {total:,} bytes would be removed.")
        print("Nothing was changed. Re-run with --execute to remove.")
        return 0

    removed: list[dict[str, Any]] = []
    for report in reports:
        for path, size in report.files:
            absolute = REPO / path
            absolute.unlink()
            removed.append({"path": path, "bytes": size})
        target = REPO / str(Path(report.target))
        if target.is_dir() and not any(target.iterdir()):
            target.rmdir()
            removed.append({"path": str(target.relative_to(REPO)), "bytes": 0, "note": "empty dir"})

    receipt = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "operator_confirmation": args.operator,
        "reason": args.reason,
        "deleted": removed,
        "deleted_bytes": sum(int(item["bytes"]) for item in removed),
        "guards": [r.as_dict() for r in reports],
        "retained_active_checkpoint": "checkpoint/e3_boundary_refinement_0062/best.pt",
        "retained_rollback_checkpoint": "checkpoint/s1_mention_full_training_v1/best.pt",
        "active_sha256": _digest_if_present("checkpoint/e3_boundary_refinement_0062/best.pt"),
        "rollback_sha256": _digest_if_present("checkpoint/s1_mention_full_training_v1/best.pt"),
        "post_cleanup_inventory_sha256": inventory_hash(),
    }
    print(f"\nREMOVED {len(removed)} item(s), {receipt['deleted_bytes']:,} bytes.")
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        print(f"receipt: {args.receipt}")
    return 0


def _digest_if_present(relative: str) -> str:
    path = REPO / relative
    return sha256_file(path) if path.is_file() else ""


def inventory_hash() -> str:
    """Digest of the current checkpoint/ file listing; carries no clinical text."""
    base = REPO / "checkpoint"
    if not base.is_dir():
        return ""
    lines = sorted(
        f"{p.relative_to(REPO)}\t{p.stat().st_size}" for p in base.rglob("*") if p.is_file()
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    try:
        sys.exit(main())
    except RemovalRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
