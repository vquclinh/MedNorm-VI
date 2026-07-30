"""Synthetic tests for Audit 0011 full-pipeline contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

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
from mednorm_vi.mention_factory.adapters import (
    AnchoredQwenProposer,
    CheckpointManifest,
    MissingCheckpointError,
    NeuralExpertAdapter,
    anchor_substring,
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
    validate_teacher_contract(
        TeacherGenerationContract("t1", "prompt", "canonical_annotations")
    )
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


def test_missing_checkpoint_and_anchored_qwen_contract(tmp_path: Path) -> None:
    missing = NeuralExpertAdapter(
        "vihealthbert",
        CheckpointManifest("vihealthbert", "span_type", str(tmp_path / "missing")),
        ("MEDICATION",),
    )
    try:
        missing.propose("1", "abc")
    except MissingCheckpointError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("missing checkpoint must fail")

    ckpt = tmp_path / "qwen"
    ckpt.mkdir()
    qwen = AnchoredQwenProposer(
        "qwen",
        CheckpointManifest("qwen", "anchored", str(ckpt)),
        ("MEDICATION",),
    )
    assert anchor_substring("abc abc", "abc", occurrence=1) == (4, 7)
    proposals = qwen.from_anchored_substrings("d1", "uống aspirin", (("aspirin", "MEDICATION"),))
    assert proposals[0].text == "aspirin"
    assert proposals[0].proposed_types == ("THUỐC",)


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
    upgraded.write_text(
        json.dumps({"labels": ["A", "B"], "kb_versions": ["v2"]}), encoding="utf-8"
    )
    report = compare_task_descriptors(current, upgraded)
    assert "labels" in report.changed_keys
    assert "kb_versions" in report.changed_keys
    assert "resolver_and_decoders" in report.requires_retraining
