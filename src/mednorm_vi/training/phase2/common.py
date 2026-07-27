"""Common deterministic helpers for Phase-2 operator artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MUTABLE_REVISIONS = {"", "main", "master", "latest", "HEAD"}
UNPINNED_PREFIXES = ("UNPINNED", "UNAVAILABLE", "CHECK_IN_MANIFEST")


class Phase2ReadinessError(ValueError):
    """Raised when a Phase-2 readiness contract is violated."""


def canonical_json_bytes(payload: Mapping[str, Any] | list[Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(payload: Mapping[str, Any] | list[Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Phase2ReadinessError(f"{path} must contain a JSON object")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise Phase2ReadinessError(f"{path}:{line_number} is not a JSON object")
            yield payload


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def is_immutable_revision(value: str, *, allow_local_architecture: bool = False) -> bool:
    if allow_local_architecture and value.startswith("mednorm-"):
        return True
    if value in MUTABLE_REVISIONS:
        return False
    if any(value.startswith(prefix) for prefix in UNPINNED_PREFIXES):
        return False
    return bool(HEX40.fullmatch(value))


def validate_hex_digest(value: str, *, field_name: str) -> None:
    if not HEX64.fullmatch(value):
        raise Phase2ReadinessError(f"{field_name} must be a 64-hex SHA-256 digest")


def privacy_safe_group_id(source_group: str) -> str:
    return hashlib.sha256(source_group.encode("utf-8")).hexdigest()[:16]


def stable_config_hash_from_file(path: str | Path) -> str:
    return sha256_file(path)


def get_repository_commit(repo_dir: str | Path) -> str:
    """Return the current commit in a checkout.

    Used by notebooks after checkout/bootstrap. The helper does not mutate the
    repository and raises clearly if Git is unavailable.
    """
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_dir),
        check=True,
        text=True,
        capture_output=True,
    )
    commit = result.stdout.strip()
    if not HEX40.fullmatch(commit):
        raise Phase2ReadinessError("resolved repository commit is not a 40-hex SHA")
    return commit


__all__ = [
    "HEX40",
    "HEX64",
    "MUTABLE_REVISIONS",
    "Phase2ReadinessError",
    "canonical_json_sha256",
    "get_repository_commit",
    "is_immutable_revision",
    "iter_jsonl",
    "privacy_safe_group_id",
    "read_json",
    "sha256_file",
    "stable_config_hash_from_file",
    "validate_hex_digest",
    "write_json",
    "write_jsonl",
]
