"""Governed model acquisition and the GLiNER rejection record (Audit 0061).

The acquisition tool is the reason this repository was finally willing to download a
model: every previous milestone declined because nothing could prove *what* had arrived.
These tests hold the fail-closed behaviour that makes the proof meaningful — a tool that
degrades gracefully when it cannot verify something is worse than no tool at all, because
it produces an artifact that looks governed and is not.

They also pin the outcome: GLiNER was acquired, evaluated and **rejected** on governed
validation. Recording a rejection is as important as recording an acceptance; without it
the next milestone would be tempted to re-run the same experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mednorm_vi.model_registry.acquire import (
    ARTIFACT_KINDS,
    KIND_MODEL,
    KIND_TOKENIZER,
    AcquiredModel,
    AcquisitionFailed,
    FileRecord,
    count_parameters,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "configs" / "models" / "candidate_model_registry.yaml"

GLINER_ID = "urchade/gliner_multi-v2.1"
GLINER_REV = "443d26d654e0324125a96bebd8e796c14ff2efe6"
GLINER_PARAMS = 288_949_504
GLINER_DIR = REPO / "models" / "mention" / "gliner_multi-v2.1" / GLINER_REV


# --- fail-closed behaviour ----------------------------------------------------


def test_an_unknown_artifact_kind_is_refused(tmp_path: Path) -> None:
    from mednorm_vi.model_registry.acquire import acquire_model

    with pytest.raises(AcquisitionFailed, match="unknown artifact kind"):
        acquire_model(model_id="x/y", root=tmp_path, kind="something-else")


def test_the_artifact_kinds_are_exactly_model_and_tokenizer() -> None:
    """A companion tokenizer carries no weights; a model may carry no tokenizer.

    Verifying both against one shape would reject one of them, so the caller declares
    which it is and verification follows the declaration.
    """
    assert set(ARTIFACT_KINDS) == {KIND_MODEL, KIND_TOKENIZER}


def test_count_parameters_refuses_an_object_without_parameters() -> None:
    with pytest.raises(AcquisitionFailed, match="no .parameters"):
        count_parameters(object())


def test_count_parameters_refuses_a_zero_count() -> None:
    class _Empty:
        def parameters(self):  # noqa: ANN202
            return iter(())

    with pytest.raises(AcquisitionFailed, match="zero"):
        count_parameters(_Empty())


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "a.bin"
    path.write_bytes(b"mednorm" * 1000)
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_an_acquired_model_serializes_its_full_provenance() -> None:
    model = AcquiredModel(
        model_id=GLINER_ID,
        pinned_revision=GLINER_REV,
        license_id="apache-2.0",
        local_path="models/x",
        files=(FileRecord("model.safetensors", 10, "ab" * 32),),
        total_bytes=10,
        parameter_count=GLINER_PARAMS,
        parameter_count_status="VERIFIED",
    )
    payload = model.as_dict()
    for key in (
        "model_id",
        "pinned_revision",
        "license",
        "source_url",
        "local_path",
        "files",
        "total_bytes",
        "parameter_count",
        "tokenizer_identity",
    ):
        assert key in payload


# --- the acquired artifact ----------------------------------------------------


@pytest.mark.skipif(not GLINER_DIR.is_dir(), reason="GLiNER not acquired locally")
def test_the_acquired_gliner_is_revision_addressed_and_pickle_free() -> None:
    assert GLINER_DIR.name == GLINER_REV, "the directory must be the immutable revision"
    files = {p.name for p in GLINER_DIR.iterdir() if p.is_file()}
    assert "model.safetensors" in files
    assert not any(f.endswith(".bin") for f in files), (
        "only safetensors were acquired: a pickle checkpoint would fall under the "
        "Audit-0056a trusted-load policy for no benefit"
    )
    config = json.loads((GLINER_DIR / "gliner_config.json").read_text(encoding="utf-8"))
    assert config["model_name"] == "microsoft/mdeberta-v3-base"


# --- the rejection record -----------------------------------------------------


def test_the_registry_records_gliner_as_verified_but_not_deployed() -> None:
    doc = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    entry = next(c for c in doc["components"] if c["component_id"] == "e6_gliner_open_type")
    assert entry["model_id"] == GLINER_ID
    assert entry["pinned_revision"] == GLINER_REV
    assert entry["total_parameters"] == GLINER_PARAMS
    assert entry["parameter_count_verified"] is True
    assert entry["parameter_count_method"] == "counted_from_checkpoint"
    # Researched, measured, and deliberately not deployed.
    assert entry["status"] == "EXCLUDED_BY_ABLATION"
    assert entry["loaded_at_inference"] is False


def test_gliner_does_not_enter_the_deployment_budget() -> None:
    """A rejected model must not consume budget, however well verified it is."""
    from mednorm_vi.governance.parameter_budget import load_candidate_registry

    registry = load_candidate_registry(REGISTRY)
    entry = next(c for c in registry.components if c.component_id == "e6_gliner_open_type")
    assert entry.total_parameters == GLINER_PARAMS
    assert not entry.loaded_at_inference


def test_the_deployment_ledger_still_reflects_e3_only() -> None:
    from mednorm_vi.governance.parameter_budget import (
        MAX_DEPLOYMENT_PARAMETERS,
        compute_deployment_budget,
        load_candidate_registry,
        load_deployment_selection,
    )

    registry = load_candidate_registry(REGISTRY)
    manifest_name, selected = load_deployment_selection(
        REPO / "configs" / "models" / "deployment_budget_template.yaml"
    )
    assert manifest_name == "template_minimal_verified"
    assert "e6_gliner_open_type" not in selected
    report = compute_deployment_budget(registry, selected)
    assert report.total_loaded_parameters < MAX_DEPLOYMENT_PARAMETERS
    assert report.within_budget
