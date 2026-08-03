"""Pinned model acquisition for GraphCENT (0080).

The first real Colab run downloaded the retrievers successfully and then wrote a manifest
whose revision fields said nothing. The weights were fine; the *record* of which weights was
not. That is a reproducibility failure: a submission cannot be re-created from a manifest
that names no commit, and a later re-download of the same floating branch may not even be
the same model.

The order here is deliberate:

1. resolve the exact commit SHA from the hub **before** downloading anything,
2. download at that SHA, so acquisition itself is pinned rather than annotated afterwards,
3. record the SHA everywhere the run's identity is stored.

A previously written manifest is consulted first. If it already carries an immutable
revision for a model, that revision is reused - a second run reproduces the first rather
than silently drifting to whatever HEAD has become. Re-resolving is what makes "reproducible"
mean nothing.

Network access lives behind two injected callables so the whole path is testable offline.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .registry import RetrieverSpec, RevisionNotPinned, is_hub_revision

MANIFEST_NAME = "model-manifest.json"


def log(message: str) -> None:
    print(f"[mednorm] {message}", flush=True)


def read_pinned_revisions(model_root: Path) -> dict[str, str]:
    """Hub commits already recorded for this model root.

    Unreadable or floating entries are treated as absent, matching the original Colab
    failure mode. Conflicting immutable entries are refused because there is no safe way to
    infer which snapshot was intended.
    """
    path = Path(model_root) / MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, str] = {}

    def remember(key: str, revision: str, source: str) -> None:
        if not key or not is_hub_revision(revision):
            return
        if key in out and out[key] != revision:
            raise RevisionNotPinned(
                f"{key}: {MANIFEST_NAME} contains conflicting pinned revisions "
                f"{out[key]} and {revision} ({source}). Refusing to choose one."
            )
        out[key] = revision

    for row in payload.get("retrievers") or []:
        remember(str(row.get("key", "")), str(row.get("revision", "")), "retrievers")
    for key, revision in (payload.get("resolved_revisions") or {}).items():
        remember(str(key), str(revision), "resolved_revisions")
    return out


def _default_info(repo_id: str, revision: str | None) -> Any:
    from huggingface_hub import model_info

    return model_info(repo_id, revision=revision)


def resolve_revision(
    spec: RetrieverSpec,
    *,
    cached: str = "",
    info_fn: Callable[[str, str | None], Any] = _default_info,
) -> str:
    """The immutable commit SHA for one model. Fails closed rather than guessing.

    `cached` wins when it is already immutable: reproducing an earlier run matters more
    than tracking the latest upload. Otherwise the hub is asked, and what comes back must
    be a commit SHA - a branch name is refused, because it is exactly the thing that
    cannot identify a set of weights.
    """
    if is_hub_revision(cached):
        log(f"{spec.key}: reusing pinned revision {cached}")
        return cached
    if is_hub_revision(spec.revision):
        return spec.revision
    requested = spec.revision or None
    try:
        info = info_fn(spec.repo_id, requested)
    except Exception as error:  # noqa: BLE001 - any hub failure is the same failure here
        raise RevisionNotPinned(
            f"{spec.key}: could not resolve a commit SHA for {spec.repo_id!r}: {error}"
        ) from error
    sha = str(getattr(info, "sha", "") or "")
    if not is_hub_revision(sha):
        raise RevisionNotPinned(
            f"{spec.key}: {spec.repo_id!r} returned revision {sha!r}, which is not an "
            "immutable commit SHA. Acquisition is refused: an unpinned model cannot be "
            "reproduced."
        )
    log(f"{spec.key}: resolved {spec.repo_id} -> {sha}")
    return sha


def acquire(
    specs: Sequence[RetrieverSpec],
    model_root: Path,
    *,
    info_fn: Callable[[str, str | None], Any] = _default_info,
    download_fn: Callable[..., str] | None = None,
) -> list[RetrieverSpec]:
    """Resolve, then download at the resolved SHA. Returns specs carrying their revision.

    Disabled retrievers are returned untouched and are never resolved or fetched: they are
    not deployed, so they have no provenance to record.

    `snapshot_download` is content-addressed and skips files already present, so calling it
    for an unchanged pin re-verifies the local directory instead of re-downloading it.
    """
    downloader = download_fn or _default_download
    cached = read_pinned_revisions(model_root)
    resolved: list[RetrieverSpec] = []
    for spec in specs:
        if not spec.enabled:
            log(f"{spec.key}: disabled in the profile - not resolved, not downloaded")
            resolved.append(spec)
            continue
        revision = resolve_revision(spec, cached=cached.get(spec.key, ""), info_fn=info_fn)
        target = Path(model_root) / spec.key
        downloader(
            repo_id=spec.repo_id, revision=revision, local_dir=str(target)
        )
        log(f"{spec.key}: {spec.repo_id}@{revision} -> {target}")
        resolved.append(replace(spec, revision=revision))
    return resolved


def _default_download(repo_id: str, revision: str, local_dir: str) -> str:
    from huggingface_hub import snapshot_download

    return str(
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)
    )


def file_digest(path: Path) -> str:
    """`sha256:<hex>` identity for a local checkpoint that has no hub revision."""
    import hashlib

    from .registry import LOCAL_DIGEST_PREFIX

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"{LOCAL_DIGEST_PREFIX}{digest.hexdigest()}"


__all__ = [
    "MANIFEST_NAME",
    "acquire",
    "file_digest",
    "read_pinned_revisions",
    "resolve_revision",
]
