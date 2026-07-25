"""Repository-checkout helper tests (Audit 0019). Uses local ``file://`` repos only."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mednorm_vi.reproducibility import (
    RepositoryCheckoutError,
    checkout_repository,
)

REPO = Path(__file__).resolve().parents[2]
REPO_URL = f"file://{REPO}"


def _head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]).decode().strip()


@pytest.mark.parametrize("url", ["", "   ", "<repo-url>", "https://example.com/<fill>",
                                 "https://github.com/YOUR_ORG/repo.git"])
def test_rejects_placeholder_or_empty_url(url: str, tmp_path: Path) -> None:
    with pytest.raises(RepositoryCheckoutError):
        checkout_repository(url, tmp_path / "co", "main")


@pytest.mark.parametrize("ref", ["", "<fill: reviewed repo HEAD>"])
def test_rejects_placeholder_or_empty_ref(ref: str, tmp_path: Path) -> None:
    with pytest.raises(RepositoryCheckoutError):
        checkout_repository(REPO_URL, tmp_path / "co", ref)


def test_checks_out_and_resolves_real_commit(tmp_path: Path) -> None:
    result = checkout_repository(REPO_URL, tmp_path / "co", "HEAD")
    assert result.resolved_commit == _head()
    assert len(result.resolved_commit) == 40
    assert Path(result.src_dir).is_dir()
    assert Path(result.adapter_file).is_file()
    assert result.adapter_file.endswith("data_engine/vietmed_ner.py")
    assert result.requested_ref == "HEAD"
    assert any(cmd.startswith("git clone") for cmd in result.commands)


def test_checkout_by_explicit_commit_sha(tmp_path: Path) -> None:
    sha = _head()
    result = checkout_repository(REPO_URL, tmp_path / "co", sha)
    assert result.resolved_commit == sha
    assert result.resolved_short == sha[:12]


def test_clean_existing_false_refuses_existing_dir(tmp_path: Path) -> None:
    target = tmp_path / "co"
    target.mkdir()
    with pytest.raises(RepositoryCheckoutError, match="already exists"):
        checkout_repository(REPO_URL, target, "HEAD", clean_existing=False)


def test_clean_existing_true_replaces_dir(tmp_path: Path) -> None:
    target = tmp_path / "co"
    target.mkdir()
    (target / "stale.txt").write_text("stale", encoding="utf-8")
    result = checkout_repository(REPO_URL, target, "HEAD", clean_existing=True)
    assert not (target / "stale.txt").exists()
    assert result.resolved_commit == _head()


def test_fails_on_unreachable_repository(tmp_path: Path) -> None:
    with pytest.raises(RepositoryCheckoutError):
        checkout_repository(f"file://{tmp_path / 'nope'}", tmp_path / "co", "HEAD")


def test_fails_when_ref_missing(tmp_path: Path) -> None:
    with pytest.raises(RepositoryCheckoutError):
        checkout_repository(REPO_URL, tmp_path / "co", "refs/heads/definitely-not-a-branch")


def test_fails_when_checkout_lacks_expected_layout(tmp_path: Path) -> None:
    """A repo without src/mednorm_vi must be rejected, not silently accepted."""
    bare = tmp_path / "other"
    bare.mkdir()
    subprocess.run(["git", "init", "-q", str(bare)], check=True)
    (bare / "README.md").write_text("not mednorm", encoding="utf-8")
    subprocess.run(["git", "-C", str(bare), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(bare), "-c", "user.email=t@e", "-c", "user.name=t",
                    "commit", "-qm", "init"], check=True)
    with pytest.raises(RepositoryCheckoutError, match="missing"):
        checkout_repository(f"file://{bare}", tmp_path / "co", "HEAD")
