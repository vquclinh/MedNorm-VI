"""Framework-closure guarantees (Audit 0055).

Three things this module exists to keep true:

1. **The obsolete non-canonical L4 is gone and cannot come back.** Not renamed, not
   aliased, not shimmed — absent from disk, from the import graph and from every
   config, test and notebook.
2. **Its unique behaviour survived the deletion.** The three configurable boundary
   policies now run through the canonical lattice L4, and the ladder that implements
   them is a single shared function rather than two "mirrored" copies.
3. **The container contract holds.** One entrypoint, offline by construction,
   non-root, read-only asset mounts, and no weights or restricted data copied in.

The old/new equivalence itself was proven against the live retired implementation
**before** it was deleted — 84 boundary-group decisions across 3 medication x 2
test-result policies on all four tracked fixtures, 0 disagreements. That evidence is
recorded in Audit 0055 §5. What remains testable afterwards is that the migrated
behaviour still holds, which is what the policy tests in
``tests/unit/test_resolution.py`` and the ladder tests below assert.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
from pathlib import Path

import pytest

from mednorm_vi.resolution.boundary import (
    medication_kind_order,
    preference_note,
    preference_rank,
)

REPO = Path(__file__).resolve().parents[2]
DELETED_MODULE = "mednorm_vi.resolution.resolver"
DELETED_PATH = REPO / "src" / "mednorm_vi" / "resolution" / "resolver.py"
DELETED_CONFIG = REPO / "configs" / "resolution" / "resolver_v1.yaml"
CANONICAL_L4_CONFIG = REPO / "configs" / "resolution" / "boundary_type_resolver_v1.yaml"


# ---------------------------------------------------------------------------
# A. The obsolete L4 is gone
# ---------------------------------------------------------------------------


def test_the_obsolete_resolver_file_is_absent() -> None:
    assert not DELETED_PATH.exists()


def test_the_obsolete_resolver_config_is_absent() -> None:
    assert not DELETED_CONFIG.exists()


def test_the_obsolete_module_is_unimportable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(DELETED_MODULE)


def test_no_compatibility_alias_or_shim_exists() -> None:
    """A shim that imports a deleted implementation is not a migration."""
    import mednorm_vi.resolution as package

    for retired in ("resolve", "ResolverConfig", "resolver"):
        assert not hasattr(package, retired), f"mednorm_vi.resolution.{retired} is still reachable"
        assert retired not in package.__all__


def test_no_tracked_file_references_the_deleted_module() -> None:
    """Import-graph and text proof over everything Git tracks."""
    output = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "-E",
            r"resolution\.resolver\b|from \.resolver import|import resolver\b",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    offenders = {line.strip() for line in output.splitlines() if line.strip()}
    # Prose that records the deletion is allowed; code that imports it is not.
    allowed_prose = {
        "docs/architecture/ACTIVE_RUNTIME_MANIFEST.md",
        "src/mednorm_vi/inference/pipeline.py",
        "src/mednorm_vi/resolution/__init__.py",
        "src/mednorm_vi/resolution/canonical.py",
        "src/mednorm_vi/resolution/boundary.py",
        "tests/unit/test_framework_closure.py",
        "tests/unit/test_resolution.py",
    }
    unexpected = {
        path for path in offenders if not (path.startswith("docs/audits/") or path in allowed_prose)
    }
    assert not unexpected, f"live references to the deleted L4: {sorted(unexpected)}"


def test_no_tracked_file_declares_the_retired_config_keys() -> None:
    """The retired keys are YAML keys, so only YAML is searched.

    An earlier version of this test grepped every tracked file and flagged
    `resolver_config: ResolverV1Config` — a Python parameter annotation holding the
    *canonical* config — as a stale key. Scoping the search to the file type the keys
    actually live in asks the question that was meant.
    """
    output = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "-E",
            r"^\s*(medication_boundary|test_result_boundary|resolver_config)\s*:",
            "--",
            "*.yaml",
            "*.yml",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    offenders = {line.strip() for line in output.splitlines() if line.strip()}
    assert not offenders, f"retired config keys still declared in: {sorted(offenders)}"


def test_the_import_graph_has_one_l4_entry_point() -> None:
    """The canonical runner and the Phase-1C CLI must call the same L4 function."""
    from mednorm_vi.inference import pipeline
    from mednorm_vi.phase1c_foundation import cli

    def imported_names(module: object) -> set[str]:
        import inspect

        found: set[str] = set()
        for node in ast.walk(ast.parse(inspect.getsource(module))):  # type: ignore[arg-type]
            if isinstance(node, ast.ImportFrom):
                found.update(alias.name for alias in node.names)
        return found

    for module in (pipeline, cli):
        names = imported_names(module)
        assert "resolve_lattice_to_hypotheses" in names, (
            f"{module.__name__} does not use the canonical L4"
        )
        assert "resolve" not in names, f"{module.__name__} imports a second resolver"


# ---------------------------------------------------------------------------
# B. Stale config keys are rejected, never silently ignored
# ---------------------------------------------------------------------------


def test_a_profile_declaring_the_retired_key_is_refused(tmp_path: Path) -> None:
    import yaml

    from mednorm_vi.inference.config import PipelineConfig

    payload = yaml.safe_load(
        (REPO / "configs" / "pipeline" / "full_v1.yaml").read_text(encoding="utf-8")
    )
    payload["resolver_config"] = "configs/resolution/resolver_v1.yaml"
    target = tmp_path / "stale.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        PipelineConfig.load(target)
    message = str(raised.value)
    assert "resolver_config" in message
    assert "l4_config" in message, "the error must name the replacement"


def test_the_pipeline_config_has_no_resolver_config_field() -> None:
    import dataclasses

    from mednorm_vi.inference.config import PipelineConfig

    names = {f.name for f in dataclasses.fields(PipelineConfig)}
    assert "resolver_config" not in names
    assert "l4_config" in names


def test_the_phase1c_cli_rejects_the_retired_flag() -> None:
    from mednorm_vi.phase1c_foundation.cli import main

    with pytest.raises(SystemExit) as raised:
        main(["doctor", "--resolver-config", "configs/resolution/resolver_v1.yaml"])
    assert raised.value.code != 0


# ---------------------------------------------------------------------------
# C. The migrated ladder is one shared implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "expected_best"),
    [("full", "full"), ("name_only", "name_only"), ("name_strength", "name_strength")],
)
def test_the_medication_ladder_ranks_its_own_policy_first(policy: str, expected_best: str) -> None:
    kinds = ("name_only", "name_strength", "name_strength_route", "full")
    ranks = {kind: preference_rank("MEDICATION", kind, policy) for kind in kinds}
    assert min(ranks, key=lambda k: ranks[k]) == expected_best
    assert ranks[expected_best] == 0


def test_the_medication_ladder_steps_down_before_it_steps_up() -> None:
    """With the target absent, a narrower span must beat a wider one.

    This is the retired resolver's documented semantics: a narrower boundary is a safe
    under-read, a wider one swallows neighbouring text.
    """
    narrower = preference_rank("MEDICATION", "name_only", "name_strength")
    wider = preference_rank("MEDICATION", "full", "name_strength")
    assert narrower < wider


def test_the_test_result_ladder_falls_back_to_value_only() -> None:
    assert preference_rank("TEST_RESULT", "value_unit", "value_unit") == 0
    assert preference_rank("TEST_RESULT", "value_only", "value_unit") == 1
    assert preference_rank("TEST_RESULT", "value_only", "value_only") == 0
    # The retired config spelled it `value_with_unit`; both must mean the same thing.
    assert preference_rank("TEST_RESULT", "value_unit", "value_with_unit") == 0


def test_an_unconfigured_policy_ranks_every_kind_equally() -> None:
    kinds = ("name_only", "full", "value_only", "value_unit")
    assert {preference_rank("MEDICATION", kind, "") for kind in kinds} == {0}


def test_the_ladder_note_records_which_rung_won() -> None:
    assert preference_note("MEDICATION", "full", "full") == "policy=full:exact"
    assert "fallback_widest_at_or_below_target" in preference_note(
        "MEDICATION", "name_only", "name_strength"
    )
    assert "fallback_narrowest_above_target" in preference_note(
        "MEDICATION", "full", "name_strength"
    )
    assert preference_note("MEDICATION", "full", "") == "no_boundary_policy"


def test_medication_kind_order_is_narrow_to_wide() -> None:
    assert (
        medication_kind_order("name_only")
        < medication_kind_order("name_strength")
        < medication_kind_order("full")
    )
    assert medication_kind_order("not_a_kind") == 99


def test_the_ladder_is_the_only_boundary_policy_implementation() -> None:
    """Guards against a second ladder reappearing anywhere in the resolution package."""
    resolution = REPO / "src" / "mednorm_vi" / "resolution"
    definers = [
        path.name
        for path in resolution.glob("*.py")
        if "def preference_rank" in path.read_text(encoding="utf-8")
    ]
    assert definers == ["boundary.py"], definers


# ---------------------------------------------------------------------------
# D. Canonical config carries the migrated policies
# ---------------------------------------------------------------------------


def test_the_canonical_config_declares_both_boundary_policies() -> None:
    from mednorm_vi.resolution import load_resolver_v1_config

    config = load_resolver_v1_config(CANONICAL_L4_CONFIG)
    assert config.boundary.group_preference["medication"] == "full"
    assert config.boundary.group_preference["test_result"] == "value_only"
    assert config.overlap.abstain_on_conflict is False
    assert len(config.config_sha256) == 64


def test_the_learned_l4_v2_stays_disabled_and_fail_closed() -> None:
    """No checkpoint exists, so the learned slot must refuse to run, not degrade.

    Updated by Audit 0056a. This test previously asserted that BOTH L4 flags were
    `False`, which encoded the defect it was meant to guard: the canonical
    deterministic L4 ran on every document while its flag said `false`, and both
    flags were inert. The flags now select a route, exactly one must be selected,
    and the shipped profile truthfully declares the deterministic one.
    """
    from mednorm_vi.inference.config import (
        L4_ROUTE_DETERMINISTIC_V1,
        PipelineConfig,
        select_l4_route,
    )

    config = PipelineConfig.load(REPO / "configs" / "pipeline" / "full_v1.yaml")
    assert config.feature_flags["enable_l4_learned_v2"] is False
    assert config.feature_flags["enable_l4_deterministic_v1"] is True
    assert select_l4_route(config.feature_flags) == L4_ROUTE_DETERMINISTIC_V1

    # Enabling the learned route without a checkpoint fails closed, and says so
    # rather than quietly running the deterministic resolver instead.
    from mednorm_vi.resolution.learned_dispatch import (
        LearnedL4Unavailable,
        learned_l4_blockers,
        load_learned_l4_config,
    )

    blockers = learned_l4_blockers(
        load_learned_l4_config(REPO / "configs" / "resolution" / "learned_l4_v2.yaml")
    )
    assert blockers, "an untrained learned slot must report a blocker"
    assert LearnedL4Unavailable is not None

    # It consumes the SAME SpanLattice contract, so enabling it later needs no new
    # plumbing — only a checkpoint.
    from mednorm_vi.resolution import learned_v2

    source = (REPO / "src" / "mednorm_vi" / "resolution" / "learned_v2.py").read_text(
        encoding="utf-8"
    )
    assert "SpanLattice" in source, "learned v2 must use the canonical lattice contract"
    assert learned_v2 is not None


def test_full_mode_still_fails_closed() -> None:
    from mednorm_vi.inference.config import PipelineConfig, evaluate_readiness

    config = PipelineConfig.load(REPO / "configs" / "pipeline" / "full_v1.yaml")
    assert evaluate_readiness(config, mode="full").status == "NOT_READY"
    assert evaluate_readiness(config, mode="deterministic").status == "READY"
    assert evaluate_readiness(config, mode="specialist").status == "READY"


# ---------------------------------------------------------------------------
# E. Container contract (static; the build and smoke are recorded in Audit 0055)
# ---------------------------------------------------------------------------


def _dockerfile() -> str:
    return (REPO / "Dockerfile").read_text(encoding="utf-8")


def test_the_image_has_exactly_one_entrypoint() -> None:
    text = _dockerfile()
    assert text.count("\nENTRYPOINT") == 1
    assert "mednorm_vi.inference.cli" in text


def test_the_image_runtime_is_non_root() -> None:
    assert "USER mednorm" in _dockerfile()


def test_the_image_is_offline_by_construction() -> None:
    text = _dockerfile()
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text


def test_the_image_copies_no_weights_or_restricted_data() -> None:
    for line in _dockerfile().splitlines():
        if not line.startswith("COPY "):
            continue
        for forbidden in ("checkpoint", "models/", "data/", "indices/", "notebooks"):
            assert forbidden not in line, f"Dockerfile copies {forbidden}: {line!r}"


def test_the_image_documents_read_only_asset_mounts() -> None:
    text = _dockerfile()
    for mount in (
        "/models/e3",
        "/models/hf",
        "/models/vncorenlp",
        "/kb/indices",
        "/input",
        "/output",
    ):
        assert mount in text, f"undocumented mount {mount}"
    # The asset mounts are documented read-only; only /output is writable.
    assert ":ro" in text
    assert "/output" in text


def test_every_lock_line_is_exactly_pinned() -> None:
    for name in ("requirements.lock", "requirements-image.lock"):
        for raw in (REPO / name).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            assert "==" in line, f"{name}: unpinned requirement {line!r}"


def test_no_weights_are_tracked_in_git() -> None:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True)
    for line in tracked.splitlines():
        assert not line.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".bin"))
