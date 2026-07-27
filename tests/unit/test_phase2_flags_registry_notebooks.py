"""Phase-2 feature flags, budget registry, notebook contracts, and no-download guards."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from mednorm_vi.evaluation.l3_l4_ablation_v2 import (
    STATUS_DISABLED,
    STATUS_UNAVAILABLE_UNTRAINED,
    plan_phase2_ablation,
)
from mednorm_vi.inference.config import DEFAULT_FEATURE_FLAGS, PipelineConfig
from mednorm_vi.model_registry.registry import ModelRole, load_registry, validate_profile_budget

REPO = Path(__file__).resolve().parents[2]


def test_phase2_default_feature_flags_preserve_measured_baseline() -> None:
    assert DEFAULT_FEATURE_FLAGS["enable_e3_vihealthbert"] is True
    assert DEFAULT_FEATURE_FLAGS["enable_l4_deterministic_v1"] is False
    assert DEFAULT_FEATURE_FLAGS["enable_e4_phobert_w2ner"] is False
    assert DEFAULT_FEATURE_FLAGS["enable_e5_xlmr_mrc"] is False
    assert DEFAULT_FEATURE_FLAGS["enable_e6_gliner"] is False
    assert DEFAULT_FEATURE_FLAGS["enable_e7_qwen_proposer"] is False
    assert DEFAULT_FEATURE_FLAGS["enable_l4_learned_v2"] is False
    config = PipelineConfig.load(REPO / "configs" / "pipeline" / "full_v1.yaml")
    assert config.feature_flags == DEFAULT_FEATURE_FLAGS


def test_phase2_ablation_reports_unavailable_untrained_instead_of_zero_predictions() -> None:
    disabled_plan = plan_phase2_ablation(DEFAULT_FEATURE_FLAGS, {"e3_vihealthbert": "missing"})
    assert any(status.status == STATUS_DISABLED for status in disabled_plan)
    enabled_flags = dict(DEFAULT_FEATURE_FLAGS)
    enabled_flags["enable_e4_phobert_w2ner"] = True
    plan = plan_phase2_ablation(
        enabled_flags,
        {"e3_vihealthbert": "missing", "e4_phobert_w2ner": ""},
    )
    assert any(
        status.arm == "E3_plus_E4_phobert_w2ner"
        and status.status == STATUS_UNAVAILABLE_UNTRAINED
        for status in plan
    )


def test_model_registry_safe_stack_counts_shared_backbones_and_has_metadata() -> None:
    roles = load_registry(REPO / "configs" / "model_registry" / "models_v1.yaml")
    budget = validate_profile_budget(roles, profile="Safe-8.85B", require_metadata=True)
    assert budget.base_parameters == 8_852_000_000
    assert budget.adapter_parameters > 0
    assert budget.within_9b
    assert not budget.metadata_errors
    assert sum(1 for role in roles if role.shared_backbone_id == "vihealthbert") >= 2


def test_model_registry_rejects_missing_revision_when_metadata_required() -> None:
    roles = (
        ModelRole(
            "bad",
            "test",
            1,
            checkpoint_hash="abc",
            enabled_profiles=("full",),
        ),
    )
    budget = validate_profile_budget(roles, profile="full", require_metadata=True)
    assert not budget.within_9b
    assert budget.metadata_errors == ("missing_model_revision:bad",)


def test_phase2_configs_forbid_auto_download_and_keep_new_experts_disabled() -> None:
    for rel in (
        "configs/mention_factory/phobert_w2ner_v1.yaml",
        "configs/mention_factory/xlmr_mrc_ner_v1.yaml",
        "configs/mention_factory/gliner_v1.yaml",
        "configs/mention_factory/qwen_proposer_v1.yaml",
        "configs/resolution/learned_l4_v2.yaml",
    ):
        doc = yaml.safe_load((REPO / rel).read_text(encoding="utf-8"))
        assert doc["enabled"] is False
        assert "download" in json.dumps(doc).lower()


def test_phase2_inference_modules_do_not_create_optimizers_or_call_backward() -> None:
    for rel in (
        "src/mednorm_vi/mention_factory/w2ner.py",
        "src/mednorm_vi/mention_factory/mrc.py",
        "src/mednorm_vi/mention_factory/gliner.py",
        "src/mednorm_vi/mention_factory/qwen_proposer.py",
        "src/mednorm_vi/resolution/learned_v2.py",
    ):
        source = (REPO / rel).read_text(encoding="utf-8")
        assert ".backward(" not in source
        assert "torch.optim" not in source
        assert "from_pretrained(" not in source


def test_phase2_notebooks_parse_and_expose_required_training_gates() -> None:
    for name in (
        "MedNorm_E4_PhoBERT_W2NER_Training.ipynb",
        "MedNorm_E5_XLMR_MRC_NER_Training.ipynb",
        "MedNorm_L4_Learned_Resolver_v2_Training.ipynb",
    ):
        doc = json.loads((REPO / "notebooks" / name).read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in doc["cells"]
            if cell["cell_type"] == "code"
        )
        for cell in doc["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), name, "exec")
        assert "RUN_FULL_TRAINING = False" in code
        assert "CONFIRM_FULL" in code
        assert "validate_corpus_hashes(CORPUS_DIR)" in code
        assert "RESUME_FROM_SMOKE_CHECKPOINT" in code
        assert "manifest.validate()" in code
        assert "validate_checkpoint_after_save_reload" in code
        assert "internal_test_accessed=False" in code
