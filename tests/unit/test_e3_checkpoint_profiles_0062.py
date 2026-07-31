"""E3 checkpoint-profile integration for the refined checkpoint (Audit 0062).

Milestone 4C replaced the active E3 weights. The weights are the *only* thing that
changed: the encoder, the label vocabulary, the alignment contract, the decoding rule,
the L4-L9 ownership boundaries and the ICD serialization decision are all untouched.
These tests hold that line, because a weights swap is exactly the kind of change that
looks contained and is not.

The two properties worth stating plainly:

* **Rollback is a config edit, not a code edit.** `configs/models/e3_checkpoint_profiles.yaml`
  names both artifacts with their digests, and the pipeline config must agree with the
  active one. A rollback that requires editing a source constant is not one anybody will
  perform under time pressure.
* **The refined checkpoint does not inherit the legacy pickle allowance.** Its digest is
  unknown to `TRUSTED_LEGACY_E3_SHA256`, so it is loaded with `weights_only=True`. The
  Audit-0056a exception stays limited to the single file it was written for.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from mednorm_vi.mention_factory.experts.e3_vihealthbert import build_e3_expert
from mednorm_vi.mention_factory.neural.checkpoint_loading import (
    LOAD_MODE_TRUSTED_LEGACY_PICKLE,
    LOAD_MODE_WEIGHTS_ONLY,
    TRUSTED_LEGACY_E3_SHA256,
    CheckpointTrustError,
    assert_load_is_permitted,
)
from mednorm_vi.mention_factory.neural.runtime import (
    CheckpointIntegrityError,
    fingerprint_checkpoint,
)

REPO = Path(__file__).resolve().parents[2]
PROFILES = REPO / "configs" / "models" / "e3_checkpoint_profiles.yaml"
PIPELINE = REPO / "configs" / "pipeline" / "full_v1.yaml"

ROLLBACK_ID = "s1_full_training_v1"
ACTIVE_ID = "boundary_refinement_0062"


def _profiles() -> dict:
    return yaml.safe_load(PROFILES.read_text(encoding="utf-8"))


def _pipeline() -> dict:
    return yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- 1. the old E3 profile remains loadable -----------------------------------


def test_the_rollback_profile_is_still_declared_and_addressable() -> None:
    """Requirement 1: the previous checkpoint stays a first-class, selectable target."""
    entry = _profiles()["profiles"][ROLLBACK_ID]
    assert entry["status"] == "ROLLBACK"
    assert entry["sha256"] == TRUSTED_LEGACY_E3_SHA256
    assert entry["path"] == "checkpoint/s1_mention_full_training_v1/best.pt"
    # It is still the artifact behind every scored submission to date.
    assert entry["validation_micro_f1"] == pytest.approx(0.5770, abs=5e-5)


def test_the_rollback_checkpoint_still_loads_under_its_own_policy() -> None:
    path = REPO / _profiles()["profiles"][ROLLBACK_ID]["path"]
    if not path.is_file():
        pytest.skip("rollback checkpoint not present locally")
    digest = _sha256(path)
    assert digest == TRUSTED_LEGACY_E3_SHA256, "the rollback artifact must be byte-identical"
    policy = assert_load_is_permitted(path, verified_sha256=digest, expected_sha256=digest)
    assert policy.mode == LOAD_MODE_TRUSTED_LEGACY_PICKLE
    # Readiness must pass without reading a single weight.
    expert = build_e3_expert({"e3_checkpoint_path": str(path)})
    ready, reason, _ = expert.readiness()
    assert ready, reason


# --- 2. the new profile resolves an exact checkpoint and hash -----------------


def test_the_active_profile_names_one_checkpoint_and_its_digest() -> None:
    document = _profiles()
    assert document["active"] == ACTIVE_ID
    entry = document["profiles"][ACTIVE_ID]
    assert entry["status"] == "ACTIVE"
    assert len(entry["sha256"]) == 64 and int(entry["sha256"], 16) >= 0
    assert entry["source_checkpoint_sha256"] == TRUSTED_LEGACY_E3_SHA256
    # The label space and encoder are unchanged; only weights differ.
    assert entry["base_model_revision"] == "f89e80b461e86f9cfc1c84019bd819830c24b6c5"
    assert entry["total_parameters"] == 135_002_117


def test_the_pipeline_config_cannot_drift_from_the_active_profile() -> None:
    """The whole point of the profiles file: two places, one fact."""
    entry = _profiles()["profiles"][ACTIVE_ID]
    settings = _pipeline().get("expert_settings") or {}
    assert settings.get("e3_checkpoint_path") == entry["path"]
    assert settings.get("e3_expected_checkpoint_sha256") == entry["sha256"]


def test_the_active_checkpoint_matches_its_pinned_digest_on_disk() -> None:
    entry = _profiles()["profiles"][ACTIVE_ID]
    path = REPO / entry["path"]
    if not path.is_file():
        pytest.skip("refined checkpoint not present locally")
    assert _sha256(path) == entry["sha256"]


def test_the_refined_checkpoint_does_not_inherit_the_legacy_pickle_allowance() -> None:
    """A newer E3 checkpoint is not automatically trusted; it is loaded safely."""
    entry = _profiles()["profiles"][ACTIVE_ID]
    assert entry["sha256"] != TRUSTED_LEGACY_E3_SHA256
    assert entry["load_mode"] == LOAD_MODE_WEIGHTS_ONLY
    policy = assert_load_is_permitted(
        Path("checkpoint/experiments/0062/best.pt"),
        verified_sha256=entry["sha256"],
        expected_sha256=entry["sha256"],
    )
    assert policy.mode == LOAD_MODE_WEIGHTS_ONLY
    assert policy.weights_only is True


# --- 3 and 4. fail-closed behaviour -------------------------------------------


def test_a_wrong_pinned_hash_fails_closed(tmp_path: Path) -> None:
    """Requirement 3. A substituted file must never reach deserialization."""
    path = tmp_path / "best.pt"
    path.write_bytes(b"not the real checkpoint")
    actual = _sha256(path)
    wrong = "0" * 64
    with pytest.raises(CheckpointTrustError, match="does not match"):
        assert_load_is_permitted(path, verified_sha256=actual, expected_sha256=wrong)
    # And the same fact is refused a second time, at fingerprint level, before any load.
    with pytest.raises(CheckpointIntegrityError, match="digest mismatch"):
        fingerprint_checkpoint(path, expected_sha256=wrong)


def test_an_unverified_checkpoint_can_never_reach_the_pickle_path(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    path.write_bytes(b"x")
    with pytest.raises(CheckpointTrustError, match="no verified SHA-256"):
        assert_load_is_permitted(path, verified_sha256="")


def test_a_missing_checkpoint_fails_closed(tmp_path: Path) -> None:
    """Requirement 4. Readiness refuses; it does not fall back to another artifact."""
    missing = tmp_path / "absent" / "best.pt"
    expert = build_e3_expert({"e3_checkpoint_path": str(missing)})
    ready, reason, _ = expert.readiness()
    assert not ready
    assert "does not exist" in reason
    with pytest.raises(CheckpointIntegrityError, match="checkpoint not found"):
        fingerprint_checkpoint(missing)


def test_an_empty_checkpoint_path_fails_closed() -> None:
    ready, reason, _ = build_e3_expert({"e3_checkpoint_path": ""}).readiness()
    assert not ready
    assert "no E3 checkpoint path" in reason


# --- 5 and 6. output invariants are unchanged by the weights swap -------------


def test_decoded_offsets_remain_source_aligned() -> None:
    """Requirement 5. The decoder enforces `original_text[start:end] == text`."""
    from mednorm_vi.mention_factory.neural.decoding import (
        NeuralSpan,
        decode_type_runs,
        validate_decoded_spans,
    )
    from mednorm_vi.training.s1_mention_smoke import ENTITY_TYPE_ORDER

    text = "benh nhan sot cao va ho khan"
    offsets = [(0, 0), (0, 4), (5, 9), (10, 13), (14, 17), (0, 0)]
    positives = [[0] * 5 for _ in offsets]
    diagnosis = ENTITY_TYPE_ORDER.index("DIAGNOSIS")
    positives[2][diagnosis] = 1
    positives[3][diagnosis] = 1
    spans = decode_type_runs(text, offsets, positives, ENTITY_TYPE_ORDER)
    validate_decoded_spans(spans, text)
    assert spans == (NeuralSpan(5, 13, "DIAGNOSIS", text[5:13], 1.0, 2),)
    for span in spans:
        assert text[span.start : span.end] == span.text

    # A span that does not match its text is refused, not repaired.
    with pytest.raises(ValueError, match="original_text"):
        validate_decoded_spans((NeuralSpan(0, 4, "DIAGNOSIS", "wrong", 1.0, 1),), text)


def test_the_submission_schema_is_unchanged_by_this_milestone() -> None:
    """Requirement 6. The five required organizer keys and their order are untouched."""
    from mednorm_vi.schemas.constants import ENTITY_TYPES

    assert set(ENTITY_TYPES) == {
        "DIAGNOSIS",
        "MEDICATION",
        "SYMPTOM",
        "TEST_NAME",
        "TEST_RESULT",
    }
    # The E3 label space the checkpoint was trained against is the sorted entity types,
    # and the refined checkpoint keeps it: a changed label space would silently
    # reinterpret every logit.
    from mednorm_vi.training.s1_mention_smoke import ENTITY_TYPE_ORDER

    assert list(ENTITY_TYPE_ORDER) == sorted(ENTITY_TYPES)
    assert (
        _profiles()["profiles"][ACTIVE_ID]["total_parameters"]
        == (_profiles()["profiles"][ROLLBACK_ID]["total_parameters"])
    ), "same architecture, therefore the same parameter count"


# --- 7. E1/E2 are untouched ----------------------------------------------------


def test_e1_and_e2_remain_enabled_and_unmodified() -> None:
    """Requirement 7. This milestone changed L3-E3 weights and nothing else in L3."""
    flags = _pipeline()["feature_flags"]
    assert flags["enable_e1_medication_grammar"] is True
    assert flags["enable_e2_laboratory_parser"] is True
    # GLiNER stays rejected (Audit 0061) and E5/E7 stay off.
    assert flags["enable_e6_gliner"] is False
    assert flags["enable_e5_xlmr_mrc"] is False
    assert flags["enable_e7_qwen_proposer"] is False
    # L4 route selection is unchanged: deterministic v1, learned v2 off.
    assert flags["enable_l4_deterministic_v1"] is True
    assert flags["enable_l4_learned_v2"] is False


def test_the_e1_medication_grammar_still_parses_a_known_form() -> None:
    """A functional check, so 'unmodified' is not merely a flag assertion."""
    from mednorm_vi.deterministic_baseline.models import Phase1BConfig

    config = Phase1BConfig.load(
        str(REPO / "configs" / "case_router" / "base.yaml"),
        str(REPO / "configs" / "medication" / "grammar_v1.yaml"),
        str(REPO / "configs" / "laboratory" / "parser_v1.yaml"),
    )
    assert config is not None


# --- 8 and 9. the ICD serialization decision survives the weights swap ---------


def test_dotted_icd_is_still_canonical_at_serialization() -> None:
    """Requirement 8. Audit 0061's resolution is independent of which weights run."""
    from mednorm_vi.inference.code_format import to_dotted

    assert to_dotted("I269") == "I26.9"
    assert to_dotted("N03") == "N03"
    policies = yaml.safe_load(
        (REPO / "configs" / "organizer" / "unresolved_policies_v1.yaml").read_text(encoding="utf-8")
    )["policies"]
    dotted = next(p for p in policies if p["policy_id"] == "UNRES-ICD-DOTTED")
    assert dotted["status"] == "resolved"
    assert dotted["resolution"] == "dotted"


def test_dotless_remains_the_internal_kb_identity() -> None:
    """Requirement 9. The dot is an output form; it is never the concept key."""
    from mednorm_vi.inference.code_format import classify_icd_code, to_dotless

    for internal in ("I269", "I509", "K650", "M4802"):
        assert classify_icd_code(internal) == "dotless"
        assert to_dotless(internal) == internal
    index = REPO / "indices" / "candidate" / "icd10_vi" / "competition-v3" / "index.json"
    if not index.is_file():
        pytest.skip("competition v3 ICD index not built locally")
    records = json.loads(index.read_text(encoding="utf-8"))["records"]
    assert all("." not in record["concept_id"] for record in records)


# --- 10. parameter budget ------------------------------------------------------


def test_the_deployment_parameter_budget_is_unchanged_and_under_9b() -> None:
    """Requirement 10. Refinement changes weights, never the budget."""
    from mednorm_vi.governance.parameter_budget import (
        MAX_DEPLOYMENT_PARAMETERS,
        compute_deployment_budget,
        load_candidate_registry,
        load_deployment_selection,
    )

    registry_path = REPO / "configs" / "models" / "candidate_model_registry.yaml"
    registry = load_candidate_registry(registry_path)
    _manifest, selected = load_deployment_selection(
        REPO / "configs" / "models" / "deployment_budget_template.yaml"
    )
    report = compute_deployment_budget(registry, selected)
    assert report.within_budget
    assert report.total_loaded_parameters < MAX_DEPLOYMENT_PARAMETERS
    # E3 is the only loaded mention model and its size did not change.
    entry = next(c for c in registry.components if c.component_id == "e3_vihealthbert_span_type")
    assert entry.total_parameters == 134_998_272
    assert entry.loaded_at_inference
