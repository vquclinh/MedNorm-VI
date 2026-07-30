"""Synthetic tests for Audit 0011 full-pipeline contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from mednorm_vi.data_engine import build_dataset
from mednorm_vi.data_engine.models import TeacherGenerationContract
from mednorm_vi.data_engine.synthetic import medication_lab_fixture
from mednorm_vi.data_engine.teacher import TeacherContractError, validate_teacher_contract
from mednorm_vi.inference.config import PipelineConfig, validate_readiness
from mednorm_vi.inference.pipeline import run_input_dir
from mednorm_vi.kb.icd10.conversion.normalization import normalize_rows
from mednorm_vi.kb.icd10.conversion.row_parser import ParsedIcdRow
from mednorm_vi.kb.icd10.conversion.validation import validate_rows
from mednorm_vi.kb.indexing.builders import build_icd_index, build_rxnorm_index
from mednorm_vi.kb.indexing.retrieval import load_index, search_index
from mednorm_vi.lattice.models import EXPERT_VIHEALTHBERT, ExpertSpanProposal
from mednorm_vi.mention_factory.offsets import ProposalRejected, resolve_occurrence
from mednorm_vi.mention_factory.registry import (
    ExpertNotReady,
    ExpertRegistration,
    ExpertRegistry,
    run_registered_experts,
)
from mednorm_vi.model_registry.registry import ModelRole, validate_profile_budget
from mednorm_vi.round2.compare import compare_task_descriptors


def test_icd_conversion_deduplicates_and_validates() -> None:
    rows = (
        ParsedIcdRow("A00", "Bệnh tả", source_page=1, source_row=1),
        ParsedIcdRow("A00", "", source_page=1, source_row=2, flags=("label_not_reconstructed",)),
        ParsedIcdRow("A00.0", "Bệnh tả do Vibrio", source_page=1, source_row=3),
    )
    normalized = normalize_rows(rows, source_document_sha256="a" * 64)
    assert [row.undotted_code for row in normalized] == ["A00", "A000"]
    assert normalized[0].children == ("A000",)
    assert "duplicate_code_rows_collapsed" in normalized[0].flags
    assert validate_rows(normalized).ok


def test_data_engine_build_hash_and_teacher_contract() -> None:
    doc = medication_lab_fixture()
    result = build_dataset((doc,), folds=3)
    assert not result.errors
    assert result.manifest.document_count == 1
    assert result.manifest.annotation_count == 3
    assert result.manifest.build_hash == build_dataset((doc,), folds=3).manifest.build_hash
    validate_teacher_contract(TeacherGenerationContract("t1", "prompt", "canonical_annotations"))
    try:
        validate_teacher_contract(
            TeacherGenerationContract(
                "t2", "prompt", "canonical_annotations", stores_chain_of_thought=True
            )
        )
    except TeacherContractError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("teacher chain-of-thought storage must be rejected")


def test_icd_and_rxnorm_indexes_retrieve_from_synthetic_sources(tmp_path: Path) -> None:
    icd_csv = tmp_path / "icd.csv"
    with icd_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "supplied_code",
                "dotted_code",
                "undotted_code",
                "vietnamese_label",
                "english_label",
                "aliases",
                "chapter",
                "block",
                "parent",
                "children",
                "specificity",
                "source_page",
                "source_row",
                "source_document_sha256",
                "status",
                "flags",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "supplied_code": "A00",
                "dotted_code": "A00",
                "undotted_code": "A00",
                "vietnamese_label": "Bệnh tả",
                "english_label": "",
                "aliases": "ta",
                "chapter": "I",
                "block": "A00-A09",
                "parent": "",
                "children": "",
                "specificity": "0",
                "source_page": "1",
                "source_row": "1",
                "source_document_sha256": "a" * 64,
                "status": "active",
                "flags": "",
            }
        )
    icd_meta = build_icd_index(icd_csv, tmp_path / "icd_index", source_snapshot_id="icd-test")
    assert icd_meta.concept_count == 1
    icd_index = load_index(tmp_path / "icd_index" / "index.json")
    assert search_index(icd_index, "benh ta")[0].concept_id == "A00"

    fixture = Path("tests/fixtures/kb/rxnorm/snapshot_a")
    rx_meta = build_rxnorm_index(fixture, tmp_path / "rx_index", source_snapshot_id="rx-test")
    assert rx_meta.concept_count > 0
    rx_index = load_index(tmp_path / "rx_index" / "index.json")
    assert search_index(rx_index, "aspirin")


class _UnreadyExpert:
    """An enabled expert whose asset is absent. Constructs nothing, loads nothing."""

    expert_id = EXPERT_VIHEALTHBERT
    role = "test_missing_checkpoint"

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint

    def readiness(self) -> tuple[bool, str, str]:
        if not self.checkpoint.is_file():
            return False, "the checkpoint file does not exist", str(self.checkpoint)
        return True, "", str(self.checkpoint)

    def prepare(self) -> None:  # pragma: no cover - never reached when unready
        raise AssertionError("prepare() must not run for an unready expert")

    def propose(self, graph: object, routings: object) -> tuple[ExpertSpanProposal, ...]:
        raise AssertionError("propose() must not run for an unready expert")


def test_an_enabled_expert_without_its_asset_fails_closed_by_name(tmp_path: Path) -> None:
    """Migrated from the deleted `mention_factory.adapters` legacy path (0056a).

    The old test exercised `NeuralExpertAdapter`, a second adapter layer on a
    different proposal type that the canonical runner never consulted. The real
    contract is the L3 registry: an ENABLED expert that is not ready raises
    `ExpertNotReady` naming the expert, the role and the exact missing path — it
    never degrades to "no proposals", because an expert silently contributing
    nothing is indistinguishable from one that genuinely found nothing.
    """
    registry = ExpertRegistry()
    missing = tmp_path / "missing" / "best.pt"
    registry.register(
        ExpertRegistration(
            expert_id=EXPERT_VIHEALTHBERT,
            feature_flag="enable_e3_vihealthbert",
            role="test_missing_checkpoint",
            factory=lambda _settings: _UnreadyExpert(missing),
        )
    )

    try:
        run_registered_experts(
            graph=object(),
            routings=(),
            feature_flags={"enable_e3_vihealthbert": True},
            registry=registry,
        )
    except ExpertNotReady as raised:
        assert raised.expert_id == EXPERT_VIHEALTHBERT
        assert str(missing) in str(raised)
    else:  # pragma: no cover - defensive
        raise AssertionError("an enabled, unready expert must fail closed")

    # Disabled: never constructed, so no asset is touched and no model is loaded.
    proposals, records = run_registered_experts(
        graph=object(),
        routings=(),
        feature_flags={"enable_e3_vihealthbert": False},
        registry=registry,
    )
    assert proposals == ()
    assert [r.reason for r in records] == ["disabled_by_profile"]


def test_offsets_for_a_text_only_source_are_resolved_not_invented() -> None:
    """The canonical replacement for the legacy `anchor_substring` helper.

    A proposal-only source (spec §6, E7) returns text without coordinates, and the
    repository resolves those coordinates itself rather than trusting the source.
    An ambiguous surface form is REFUSED instead of resolved to a guess.
    """
    assert resolve_occurrence("uống aspirin", "aspirin") == (5, 12)
    # Repeated surface form, no anchor: refused rather than resolved to occurrence 0.
    repeated = "aspirin 81mg và aspirin 100mg"
    with pytest.raises(ProposalRejected):
        resolve_occurrence(repeated, "aspirin")
    # An anchor that occurs once, and contains the mention once, disambiguates it.
    assert resolve_occurrence(repeated, "aspirin", anchor="aspirin 100mg") == (16, 23)


def test_pipeline_deterministic_packaging_and_full_readiness(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text("Thuốc: aspirin 81mg\n", encoding="utf-8")
    (input_dir / "2.txt").write_text("Không ghi nhận thuốc.\n", encoding="utf-8")
    config_path = tmp_path / "pipeline.yaml"
    config_payload = {
        "l1_config": "configs/document_intelligence/base.yaml",
        "router_config": "configs/case_router/base.yaml",
        "medication_config": "configs/medication/grammar_v1.yaml",
        "laboratory_config": "configs/laboratory/parser_v1.yaml",
        "expected_documents": 2,
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "full_requires_checkpoints": ["missing-full"],
    }
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    config = PipelineConfig.load(config_path)
    assert validate_readiness(config, mode="full")
    results = run_input_dir(
        input_dir,
        output_zip=tmp_path / "output.zip",
        config=config,
        mode="deterministic",
    )
    assert len(results) == 2
    assert (tmp_path / "output.zip").is_file()


def test_model_budget_shared_backbone_and_round2_compare(tmp_path: Path) -> None:
    roles = (
        ModelRole("head-a", "span", 100, 10, shared_backbone_id="bb", enabled_profiles=("full",)),
        ModelRole("head-b", "type", 100, 20, shared_backbone_id="bb", enabled_profiles=("full",)),
    )
    budget = validate_profile_budget(roles, profile="full", limit_parameters=150)
    assert budget.base_parameters == 100
    assert budget.adapter_parameters == 30
    assert budget.within_9b

    current = tmp_path / "current.json"
    upgraded = tmp_path / "upgraded.json"
    current.write_text(json.dumps({"labels": ["A"], "kb_versions": ["v1"]}), encoding="utf-8")
    upgraded.write_text(json.dumps({"labels": ["A", "B"], "kb_versions": ["v2"]}), encoding="utf-8")
    report = compare_task_descriptors(current, upgraded)
    assert "labels" in report.changed_keys
    assert "kb_versions" in report.changed_keys
    assert "resolver_and_decoders" in report.requires_retraining
