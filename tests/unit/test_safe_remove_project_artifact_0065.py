"""The guarded artifact-removal utility (Audit 0065 §2).

Audit 0064 §2 recorded two real cleanup defects, and these tests exist so neither can
recur silently:

1. a substring grep counted a **provenance** field (`experiment_origin`) as a live
   runtime reference, which would have blocked a legitimate removal forever;
2. the script printed `ABORT` and then deleted anyway, because printing is not a guard.

The second is the dangerous one, so it is tested by exit status rather than by output:
every refusal must terminate with a nonzero code *before* anything is unlinked.

Nothing here touches a real checkpoint. The only files removed are fixtures these tests
create themselves.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "safe_remove_project_artifact.py"
ACTIVE = "checkpoint/e3_boundary_refinement_0062/best.pt"
ROLLBACK = "checkpoint/s1_mention_full_training_v1/best.pt"
SCRATCH = REPO / "runs" / "diagnostics" / "0065_removal_fixtures"


def _module():
    spec = importlib.util.spec_from_file_location("safe_remove_project_artifact", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _module()


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def fixture_file():
    """An unreferenced generated artifact, created and cleaned up by this module."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "removable_artifact.bin"
    path.write_bytes(b"generated fixture" * 64)
    yield path
    path.unlink(missing_ok=True)
    if SCRATCH.is_dir() and not any(SCRATCH.iterdir()):
        SCRATCH.rmdir()


# --- protected artifacts ------------------------------------------------------


@pytest.mark.skipif(not (REPO / ACTIVE).is_file(), reason="active checkpoint absent locally")
def test_the_active_checkpoint_cannot_be_removed() -> None:
    report = MODULE.inspect(ACTIVE)
    assert not report.passed
    failed = [name for name, ok, _d in report.checks if not ok]
    assert "not an active or rollback checkpoint (by digest)" in failed
    # Refused by DIGEST, so renaming or relocating the file changes nothing.
    detail = next(d for n, ok, d in report.checks if not ok and "digest" in n)
    assert "ACTIVE E3 checkpoint" in detail


@pytest.mark.skipif(not (REPO / ROLLBACK).is_file(), reason="rollback checkpoint absent locally")
def test_the_rollback_checkpoint_cannot_be_removed() -> None:
    report = MODULE.inspect(ROLLBACK)
    assert not report.passed
    detail = next(d for n, ok, d in report.checks if not ok and "digest" in n)
    assert "ROLLBACK E3 checkpoint" in detail


@pytest.mark.skipif(not (REPO / ACTIVE).is_file(), reason="active checkpoint absent locally")
def test_refusing_the_active_checkpoint_exits_nonzero() -> None:
    """Audit 0064 defect 2: a printed warning is not a guard. The status code is."""
    result = run_cli(ACTIVE, "--execute")
    assert result.returncode != 0, "a failed guard must terminate the process"
    assert (REPO / ACTIVE).is_file(), "the protected file must still exist"


# --- the provenance-vs-live distinction (Audit 0064 defect 1) -----------------


def test_a_provenance_only_reference_is_not_a_live_reference() -> None:
    """`experiment_origin` documents history; it does not load anything.

    This is the exact field whose substring match made the Audit-0064 guard fire on an
    artifact that nothing could load.
    """
    origin = "checkpoint/experiments/0062_e3_boundary_refinement/R3_alpha050/best.pt"
    references = MODULE.find_references(origin)
    assert references, "the profile registry should still document the origin"
    assert all(not r.live for r in references), [r.describe() for r in references]
    assert any(r.field == "experiment_origin" for r in references)


def test_the_active_load_path_is_a_live_reference() -> None:
    """The same mechanism must still catch a genuine runtime reference."""
    references = MODULE.find_references(ACTIVE)
    live = [r for r in references if r.live]
    assert live, [r.describe() for r in references]
    assert {r.field for r in live} & {"path", "e3_checkpoint_path", "active_checkpoint_path"}


def test_a_live_config_reference_blocks_removal(monkeypatch) -> None:
    """A path named by a load-bearing field is refused even when it is not protected."""
    monkeypatch.setattr(
        MODULE,
        "find_references",
        lambda relative: [MODULE.Reference("fake.yaml:profiles.x", "path", True)],
    )
    SCRATCH.mkdir(parents=True, exist_ok=True)
    victim = SCRATCH / "referenced.bin"
    victim.write_bytes(b"x")
    try:
        report = MODULE.inspect(str(victim.relative_to(REPO)))
        assert not report.passed
        assert any(not ok and "LIVE" in n for n, ok, _d in report.checks)
    finally:
        victim.unlink(missing_ok=True)
        if SCRATCH.is_dir() and not any(SCRATCH.iterdir()):
            SCRATCH.rmdir()


# --- path safety --------------------------------------------------------------


def test_a_path_outside_the_repository_is_refused() -> None:
    with pytest.raises(MODULE.RemovalRefused, match="outside the repository"):
        MODULE.inspect("/etc/hostname")


def test_a_symlink_is_refused(tmp_path: Path) -> None:
    """A link is how a bounded target becomes an unbounded one."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    link = SCRATCH / "escape_link"
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    link.symlink_to(target)
    try:
        with pytest.raises(MODULE.RemovalRefused, match="symlink"):
            MODULE.inspect(str(link.relative_to(REPO)))
        assert target.is_file(), "the symlink target must be untouched"
    finally:
        link.unlink(missing_ok=True)
        if SCRATCH.is_dir() and not any(SCRATCH.iterdir()):
            SCRATCH.rmdir()


def test_a_directory_containing_subdirectories_is_refused() -> None:
    """Only a bounded directory may be a target; recursion is never implicit."""
    report = MODULE.inspect("checkpoint")
    assert not report.passed
    assert any(not ok and "bounded" in n for n, ok, _d in report.checks)


# --- dry run vs execute -------------------------------------------------------


def test_dry_run_changes_nothing(fixture_file: Path) -> None:
    before = fixture_file.read_bytes()
    result = run_cli(str(fixture_file.relative_to(REPO)))
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert fixture_file.is_file(), "dry run must not remove anything"
    assert fixture_file.read_bytes() == before


def test_an_unreferenced_fixture_is_removed_only_with_execute(fixture_file: Path) -> None:
    relative = str(fixture_file.relative_to(REPO))
    assert run_cli(relative).returncode == 0
    assert fixture_file.is_file(), "still present after the dry run"

    receipt = SCRATCH / "receipt.json"
    result = run_cli(
        relative,
        "--execute",
        "--receipt",
        str(receipt),
        "--operator",
        "unit-test",
        "--reason",
        "generated fixture",
    )
    assert result.returncode == 0, result.stderr
    assert not fixture_file.exists(), "the fixture should be gone after --execute"
    assert receipt.is_file(), "an executed removal must leave a receipt"

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["deleted_bytes"] > 0
    assert payload["retained_active_checkpoint"] == ACTIVE
    assert payload["post_cleanup_inventory_sha256"]
    receipt.unlink(missing_ok=True)


def test_a_missing_target_is_refused_not_silently_ignored() -> None:
    report = MODULE.inspect("checkpoint/does_not_exist.pt")
    assert not report.passed
    assert any(not ok and "exists" in n for n, ok, _d in report.checks)


def test_the_cli_exits_nonzero_when_any_target_in_a_group_fails(fixture_file: Path) -> None:
    """One bad target refuses the whole group; a partial deletion is never safer."""
    result = run_cli(str(fixture_file.relative_to(REPO)), ACTIVE, "--execute")
    assert result.returncode != 0
    assert fixture_file.is_file(), "the good target must not be removed either"
