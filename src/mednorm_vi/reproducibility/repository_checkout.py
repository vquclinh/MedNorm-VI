"""Real, tested repository checkout for Colab workflows (Audit 0019).

Colab notebooks must not carry ad-hoc commented `!git clone` lines: a commented
command silently does nothing and the notebook then fails with
``ModuleNotFoundError: No module named 'mednorm_vi'``. This module performs an
actual clone/fetch/checkout via ``subprocess`` argument arrays (never shell
interpolation), validates the result, and returns the **resolved commit SHA** so
every artifact manifest records real provenance instead of the word ``main``.

Pure stdlib; no network is required by its tests (they use ``file://`` repos).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Rejects unfilled template values such as "<repo-url>" or "<fill: ...>".
_PLACEHOLDER = re.compile(r"<[^>]*>|REPLACE_ME|IMPLEMENT_ME|INSERT_|YOUR_|your-repo")

_EXPECTED_SRC = Path("src") / "mednorm_vi"
_EXPECTED_ADAPTER = Path("src") / "mednorm_vi" / "data_engine" / "vietmed_ner.py"


class RepositoryCheckoutError(RuntimeError):
    """Raised when the checkout cannot be performed or verified."""


@dataclass(frozen=True, slots=True)
class CheckoutResult:
    """Outcome of a verified checkout."""

    repo_dir: str
    repo_url: str
    requested_ref: str
    resolved_commit: str
    src_dir: str
    adapter_file: str
    commands: tuple[str, ...] = field(default_factory=tuple)

    @property
    def resolved_short(self) -> str:
        return self.resolved_commit[:12]


def _reject_placeholder(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise RepositoryCheckoutError(f"{field_name} must not be empty")
    if _PLACEHOLDER.search(value):
        raise RepositoryCheckoutError(
            f"{field_name} still contains a placeholder value: {value!r}")


def _run(args: list[str], *, cwd: Path | None = None) -> str:
    """Run a git command (argument array; no shell). Raises on failure."""
    try:
        proc = subprocess.run(
            args, cwd=str(cwd) if cwd else None, capture_output=True, text=True, check=False)
    except OSError as exc:  # pragma: no cover - git missing
        raise RepositoryCheckoutError(f"failed to execute {args[:2]}: {exc}") from exc
    if proc.returncode != 0:
        # Diagnostics only; git argument arrays contain no credentials.
        raise RepositoryCheckoutError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n"
            f"stderr: {proc.stderr.strip()[:500]}")
    return proc.stdout.strip()


def checkout_repository(
    repo_url: str,
    repo_dir: str | Path,
    repo_ref: str = "main",
    *,
    clean_existing: bool = True,
) -> CheckoutResult:
    """Clone ``repo_url`` into ``repo_dir`` and check out ``repo_ref``.

    Fails fast on placeholder/empty inputs, on any git failure, and when the
    checkout does not actually contain ``src/mednorm_vi`` and the VietMed adapter.
    ``repo_ref`` may be a branch, tag, or full commit SHA; the **resolved** commit
    is returned for artifact provenance.

    ``clean_existing=True`` removes a pre-existing ``repo_dir`` (Colab temp
    storage); ``False`` refuses to touch it and raises if it already exists.
    """
    _reject_placeholder(repo_url, "repo_url")
    _reject_placeholder(repo_ref, "repo_ref")
    target = Path(repo_dir)
    commands: list[str] = []

    if target.exists():
        if not clean_existing:
            raise RepositoryCheckoutError(
                f"{target} already exists and clean_existing=False")
        shutil.rmtree(target)
        commands.append(f"rmtree {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--no-checkout", repo_url, str(target)])
    commands.append(f"git clone --no-checkout {repo_url} {target}")

    # Fetch the requested ref explicitly so branches, tags and bare SHAs all work.
    try:
        _run(["git", "fetch", "--depth", "1", "origin", repo_ref], cwd=target)
        commands.append(f"git fetch --depth 1 origin {repo_ref}")
        checkout_target = "FETCH_HEAD"
    except RepositoryCheckoutError:
        # Some refs (e.g. a full SHA on a server that disallows single-sha fetch)
        # need the full history; fall back to a complete fetch, then checkout by ref.
        _run(["git", "fetch", "origin"], cwd=target)
        commands.append("git fetch origin")
        checkout_target = repo_ref

    _run(["git", "checkout", "--detach", checkout_target], cwd=target)
    commands.append(f"git checkout --detach {checkout_target}")

    resolved = _run(["git", "rev-parse", "HEAD"], cwd=target)
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise RepositoryCheckoutError(f"unexpected resolved commit: {resolved!r}")

    src_dir = target / _EXPECTED_SRC
    if not src_dir.is_dir():
        raise RepositoryCheckoutError(f"checkout missing {_EXPECTED_SRC} at {src_dir}")
    adapter = target / _EXPECTED_ADAPTER
    if not adapter.is_file():
        raise RepositoryCheckoutError(f"checkout missing adapter at {adapter}")

    return CheckoutResult(
        repo_dir=str(target), repo_url=repo_url, requested_ref=repo_ref,
        resolved_commit=resolved, src_dir=str(target / "src"),
        adapter_file=str(adapter), commands=tuple(commands))


__all__ = ["CheckoutResult", "RepositoryCheckoutError", "checkout_repository"]
