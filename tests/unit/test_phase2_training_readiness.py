"""Phase-2 training readiness, artifact validation, and ablation contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mednorm_vi.evaluation.l3_l4_ablation_v2 import (
    ARM_E3_E4,
    ARM_E3_E4_E5,
    ARM_E3_ONLY,
    STATUS_EVALUABLE,
    STATUS_UNAVAILABLE_UNTRAINED,
    AblationArmStatus,
)
from mednorm_vi.lattice import ExpertSpanProposal, build_span_lattice
from mednorm_vi.lattice.models import EXPERT_PHOBERT_W2NER, EXPERT_VIHEALTHBERT
from mednorm_vi.mention_factory.mrc import TYPE_QUERY_ORDER
from mednorm_vi.mention_factory.neural.decoding import NeuralSpan
from mednorm_vi.model_registry.registry import ModelRole
from mednorm_vi.resolution.learned_v2 import (
    BOUNDARY_KEEP,
    GoldMention,
    ResolverV2Config,
    build_training_examples,
)
from mednorm_vi.training.phase2.artifacts import (
    MODE_FULL,
    REQUIRED_ARTIFACT_FILES,
    STATUS_FULLY_TRAINED,
    ArtifactValidationReport,
    Phase2TrainingManifest,
    checkpoint_payload,
    validate_e4_artifact,
    validate_e5_artifact,
    write_checkpoint_payload,
)
from mednorm_vi.training.phase2.budget import validate_phase2_validation_profile_budget
from mednorm_vi.training.phase2.common import canonical_json_sha256, sha256_file, write_json
from mednorm_vi.training.phase2.e5_mrc_training import query_hash
from mednorm_vi.training.phase2.internal_test_gate import (
    INTERNAL_TEST_AUTHORIZATION,
    evaluate_internal_test_freeze_gate,
)
from mednorm_vi.training.phase2.l4_training import (
    TYPE_ACTION_ORDER,
    L4ModelOutputs,
    l4_loss_terms,
    target_indices,
    validate_boundary_action_output,
)
from mednorm_vi.training.phase2.proposal_generation import (
    STATUS_AVAILABLE,
    FrozenExpertAvailability,
    ProposalDocument,
    build_frozen_proposal_dataset,
    write_frozen_proposal_dataset,
)
from mednorm_vi.training.phase2.proposal_generation import (
    STATUS_UNAVAILABLE_UNTRAINED as PROPOSAL_UNAVAILABLE,
)
from mednorm_vi.training.phase2.validation_ablation import (
    run_validation_ablation,
)


def _write_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "epoch": 1,
                "mode": MODE_FULL,
                "train_loss": 0.5,
                "validation_exact_f1": 0.25,
                "optimizer_steps": 3,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_valid_artifact(
    root: Path,
    *,
    expert_id: str = EXPERT_PHOBERT_W2NER,
    label_space: tuple[str, ...] = (
        "DIAGNOSIS",
        "MEDICATION",
        "SYMPTOM",
        "TEST_NAME",
        "TEST_RESULT",
    ),
    query_revision: str = "",
    query_digest: str = "",
) -> None:
    resolved_config = {
        "stage_id": "stage",
        "expert_id": expert_id,
        "mode": MODE_FULL,
        "internal_test_accessed": False,
    }
    config_sha = canonical_json_sha256(resolved_config)
    write_json(root / "resolved_config.json", resolved_config)
    write_json(root / "validation_metrics.json", {
        "validation_exact_f1": 0.25,
        "internal_test_accessed": False,
    })
    _write_history(root / "logs" / "training_history.jsonl")
    for name, epoch in (("best", 1), ("latest", 2)):
        payload = checkpoint_payload(
            expert_id=expert_id,
            mode=MODE_FULL,
            config_sha256=config_sha,
            model_revision="a" * 40,
            tokenizer_revision="b" * 40,
            query_revision=query_revision,
            parameter_count=11,
            label_space=label_space,
        )
        payload["epoch"] = epoch
        write_checkpoint_payload(root / "checkpoints" / f"{name}.pt", payload)
    checkpoint_hashes = {
        name: sha256_file(root / "checkpoints" / f"{name}.pt")
        for name in ("best", "latest")
    }
    manifest = Phase2TrainingManifest(
        stage_id="stage",
        expert_id=expert_id,
        mode=MODE_FULL,
        status=STATUS_FULLY_TRAINED,
        run_completed=True,
        interrupted_reason="",
        safe_to_resume=True,
        repository_commit="c" * 40,
        corpus_hashes={"train": "d" * 64, "validation": "e" * 64},
        data_hashes={"train": "d" * 64, "validation": "e" * 64},
        config_sha256=config_sha,
        model_id="model",
        model_revision="a" * 40,
        tokenizer_revision="b" * 40,
        query_revision=query_revision,
        query_hash=query_digest,
        seed=1,
        completed_epochs=1,
        optimizer_steps=3,
        effective_batch_size=2,
        parameter_count=11,
        checkpoint_hashes=checkpoint_hashes,
        best_metric=0.25,
        best_metric_name="validation_exact_f1",
        best_criterion="max_validation_exact_f1_governed_validation_only",
        train_split_id="train",
        validation_split_id="validation",
        internal_test_accessed=False,
        initialization_source="pinned_pretrained_base",
        label_space=label_space,
    )
    manifest.write(root / "training_manifest.json")


def test_artifact_validator_accepts_six_file_contract(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path)
    assert sorted(str(path) for path in REQUIRED_ARTIFACT_FILES)
    report = validate_e4_artifact(tmp_path)
    assert report.ok, report.failures
    assert set(report.checkpoint_hashes) == {"best", "latest"}


def test_artifact_validator_reports_multiple_failures(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path)
    (tmp_path / "validation_metrics.json").unlink()
    manifest_path = tmp_path / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["internal_test_accessed"] = True
    manifest["model_revision"] = "main"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    report = validate_e4_artifact(tmp_path)
    assert not report.ok
    assert "missing_required_file:validation_metrics.json" in report.failures
    assert "manifest_internal_test_accessed" in report.failures
    assert "model_revision_not_immutable" in report.failures


def test_e5_artifact_requires_query_hash(tmp_path: Path) -> None:
    _write_valid_artifact(
        tmp_path,
        expert_id="E5_xlmr_mrc_ner",
        label_space=TYPE_QUERY_ORDER,
        query_revision="mrc-type-queries-v1",
        query_digest=query_hash(),
    )
    assert validate_e5_artifact(tmp_path).ok
    manifest_path = tmp_path / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["query_hash"] = ""
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    assert "query_hash_missing" in validate_e5_artifact(tmp_path).failures


def _expert_proposal(document_id: str, text: str) -> ExpertSpanProposal:
    start = text.index("ho")
    return ExpertSpanProposal(
        document_id=document_id,
        start=start,
        end=len(text),
        text=text[start:],
        type_scores={"SYMPTOM": 0.8},
        local_score=0.8,
        expert_id=EXPERT_PHOBERT_W2NER,
        proposal_id="e4-1",
        original_start=start,
        original_end=len(text),
        model_revision="a" * 40,
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
    )


def test_frozen_proposal_dataset_is_deterministic_and_hashes_jsonl(tmp_path: Path) -> None:
    document = ProposalDocument("doc", "Bệnh nhân ho khan", "source-a", "train")
    availability = (
        FrozenExpertAvailability(
            expert_id=EXPERT_PHOBERT_W2NER,
            status=STATUS_AVAILABLE,
            config_sha256="c" * 64,
            checkpoint_sha256="b" * 64,
            model_revision="a" * 40,
        ),
        FrozenExpertAvailability(
            expert_id="E5_xlmr_mrc_ner",
            status=PROPOSAL_UNAVAILABLE,
            config_sha256="d" * 64,
            reason="untrained",
        ),
    )
    proposals = {"doc": (_expert_proposal("doc", document.original_text),)}
    left = build_frozen_proposal_dataset(
        (document,),
        proposals,
        split="train",
        config_sha256="e" * 64,
        expert_availability=availability,
    )
    right = build_frozen_proposal_dataset(
        (document,),
        proposals,
        split="train",
        config_sha256="e" * 64,
        expert_availability=tuple(reversed(availability)),
    )
    assert left.determinism_hash() == right.determinism_hash()
    hashes = write_frozen_proposal_dataset(tmp_path, left)
    assert hashes["proposals_jsonl_sha256"] == left.manifest.proposal_dataset_sha256
    assert left.records[0].privacy_safe_group_id != "source-a"


def test_l4_loss_shapes_offsets_and_wrong_type_targets() -> None:
    text = "Bệnh nhân suy tim"
    start = text.index("suy")
    lattice = build_span_lattice(
        "doc",
        text,
        neural_spans=(NeuralSpan(start, len(text), "SYMPTOM", "suy tim", 0.9, 2),),
    )
    examples = build_training_examples(
        lattice,
        (GoldMention("doc", start, len(text), "suy tim", "DIAGNOSIS", "group"),),
        split="train",
        source_group="group",
    )
    target = target_indices(examples[0])
    outputs = L4ModelOutputs(
        boundary_logits=(3.0, 0.0, 0.0, -1.0),
        type_logits=tuple(3.0 if item == "DIAGNOSIS" else -1.0 for item in TYPE_ACTION_ORDER),
        wrong_type_logit=1.0,
        iou_score=2.0,
        start_delta=0,
        end_delta=0,
    )
    terms = l4_loss_terms(outputs, target)
    assert terms["total_loss"] >= 0.0
    assert target.wrong_type_cost == 2.0
    assert validate_boundary_action_output(
        proposal_start=start,
        proposal_end=len(text),
        original_text=text,
        action=BOUNDARY_KEEP,
        start_delta=0,
        end_delta=0,
        config=ResolverV2Config(max_boundary_delta=12),
    ) == (start, len(text))
    with pytest.raises(ValueError):
        validate_boundary_action_output(
            proposal_start=start,
            proposal_end=len(text),
            original_text=text,
            action="OFFSET",
            start_delta=99,
            end_delta=0,
            config=ResolverV2Config(max_boundary_delta=12),
        )


def test_validation_ablation_uses_validation_only_and_unavailable_statuses() -> None:
    text = "suy tim"
    examples = (
        {
            "example_id": "ex",
            "split": "validation",
            "text": text,
            "entities": [
                {"start": 0, "end": len(text), "target_type": "DIAGNOSIS", "text": text}
            ],
        },
    )
    statuses = (
        AblationArmStatus(ARM_E3_ONLY, STATUS_EVALUABLE, "ok", (EXPERT_VIHEALTHBERT,), ()),
        AblationArmStatus(
            ARM_E3_E4,
            STATUS_UNAVAILABLE_UNTRAINED,
            "missing",
            (EXPERT_VIHEALTHBERT, EXPERT_PHOBERT_W2NER),
            ("e4",),
        ),
        AblationArmStatus(ARM_E3_E4_E5, STATUS_UNAVAILABLE_UNTRAINED, "missing", (), ()),
    )
    report = run_validation_ablation(
        examples,
        {
            ARM_E3_ONLY: {
                "ex": [
                    {
                        "start": 0,
                        "end": len(text),
                        "entity_type": "DIAGNOSIS",
                        "text": text,
                    }
                ]
            }
        },
        arm_statuses=statuses,
        config_hashes_by_arm={},
        checkpoint_hashes_by_arm={},
    )
    rows = {row["arm"]: row for row in report.as_dict()["results"]}
    assert rows[ARM_E3_ONLY]["exact_f1"] == 1.0
    assert rows[ARM_E3_E4]["status"] == STATUS_UNAVAILABLE_UNTRAINED
    with pytest.raises(ValueError):
        run_validation_ablation(
            ({**examples[0], "split": "internal_test"},),
            {},
            arm_statuses=statuses,
            config_hashes_by_arm={},
            checkpoint_hashes_by_arm={},
        )


def test_internal_test_gate_requires_authorization_and_valid_hashes() -> None:
    artifact = ArtifactValidationReport(
        artifact_dir="/tmp/art",
        expected_expert_id=EXPERT_PHOBERT_W2NER,
        expected_mode=MODE_FULL,
        ok=True,
        failures=(),
        warnings=(),
        checkpoint_hashes={"best": "a" * 64},
    )
    report = evaluate_internal_test_freeze_gate(
        artifact_reports=(artifact,),
        frozen_feature_flags={"enable_e3_vihealthbert": True},
        frozen_thresholds={"threshold": 0.5},
        config_hashes={"profile": "b" * 64},
        checkpoint_hashes={"e3": "c" * 64},
        validation_ablation_complete=True,
        validation_ablation_hash="d" * 64,
        model_revisions={"e3": "e" * 40},
        authorization="",
    )
    assert not report.ready
    assert "operator_authorization_missing_or_invalid" in report.failures
    ready = evaluate_internal_test_freeze_gate(
        artifact_reports=(artifact,),
        frozen_feature_flags={"enable_e3_vihealthbert": True},
        frozen_thresholds={"threshold": 0.5},
        config_hashes={"profile": "b" * 64},
        checkpoint_hashes={"e3": "c" * 64},
        validation_ablation_complete=True,
        validation_ablation_hash="d" * 64,
        model_revisions={"e3": "e" * 40},
        authorization=INTERNAL_TEST_AUTHORIZATION,
    )
    assert ready.ready
    assert not ready.internal_test_accessed


def test_phase2_budget_rejects_unpinned_or_missing_enabled_models() -> None:
    roles = (
        ModelRole(
            model_id="good",
            role="test",
            base_parameter_count=135,
            adapter_parameter_count=2,
            shared_backbone_id="good",
            model_revision="a" * 40,
            checkpoint_hash="b" * 64,
            local_path=str(Path(__file__)),
        ),
        ModelRole(
            model_id="bad",
            role="test",
            base_parameter_count=10,
            shared_backbone_id="bad",
            model_revision="main",
            checkpoint_hash="UNAVAILABLE_UNTRAINED",
            local_path="",
        ),
    )
    ok = validate_phase2_validation_profile_budget(
        roles,
        profile_name="phase2-validation",
        model_ids=("good",),
    )
    assert ok.ok
    bad = validate_phase2_validation_profile_budget(
        roles,
        profile_name="phase2-validation",
        model_ids=("good", "bad"),
    )
    assert not bad.ok
    assert "model_revision_not_immutable:bad" in bad.failures
    assert "checkpoint_hash_missing:bad" in bad.failures


def test_notebooks_and_git_do_not_track_runtime_weights() -> None:
    repo = Path(__file__).resolve().parents[2]
    for notebook in (
        "MedNorm_E4_PhoBERT_W2NER_Training.ipynb",
        "MedNorm_E5_XLMR_MRC_NER_Training.ipynb",
        "MedNorm_L4_Learned_Resolver_v2_Training.ipynb",
        "MedNorm_Phase2_Validation_Ablation.ipynb",
    ):
        doc = json.loads((repo / "notebooks" / notebook).read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in doc["cells"])
        assert "/content/drive/MyDrive/MedNorm-VI/artifacts/" in source
        assert "output.zip" in source
        for cell in doc["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), notebook, "exec")
    tracked = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines()
    forbidden_suffixes = (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip")
    assert not [path for path in tracked if path.endswith(forbidden_suffixes)]
