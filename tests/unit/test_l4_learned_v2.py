"""Learned L4 v2 target construction, leakage controls, and runtime contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mednorm_vi.lattice import build_span_lattice
from mednorm_vi.mention_factory.neural.decoding import NeuralSpan
from mednorm_vi.resolution.learned_v2 import (
    BOUNDARY_KEEP,
    BOUNDARY_OFFSET,
    BOUNDARY_SELECT_EXISTING,
    RESOLVER_V2_VERSION,
    GoldMention,
    ResolverV2Config,
    ResolverV2Error,
    ResolverV2UnavailableError,
    build_training_examples,
    grouped_train_validation_split,
    load_checkpoint,
    resolve_lattice,
    validate_runtime_checkpoint,
)


def _write_checkpoint(path: Path, *, wrong_type_bias: float = -4.0) -> str:
    payload = {
        "metadata": {
            "stage": "smoke",
            "expert": RESOLVER_V2_VERSION,
            "config_sha256": "a" * 64,
            "corpus_sha256": "b" * 64,
            "model_revision": "linear-smoke",
            "seed": 1,
            "git_commit": "c" * 40,
            "checkpoint_sha256": "FILLED_AFTER_WRITE",
            "parameter_count": 7,
            "train_split_id": "train-groups",
            "validation_split_id": "validation-groups",
            "internal_test_accessed": False,
        },
        "action_weights": {
            "KEEP": {"bias": 2.0},
            "DROP": {"bias": -2.0},
        },
        "type_weights": {
            "DIAGNOSIS": {"type_score_DIAGNOSIS": 3.0, "bias": 0.1},
            "SYMPTOM": {"type_score_SYMPTOM": 3.0},
            "DROP": {"bias": -2.0},
        },
        "wrong_type_weights": {"bias": wrong_type_bias},
        "thresholds": {
            "keep_threshold": 0.5,
            "abstain_margin": 0.05,
            "wrong_type_risk_threshold": 0.35,
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload["metadata"]["checkpoint_sha256"] = digest
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_learned_l4_targets_wrong_diagnosis_symptom_type_cost() -> None:
    text = "Bệnh nhân suy tim"
    start = text.index("suy")
    lattice = build_span_lattice(
        "doc",
        text,
        neural_spans=(NeuralSpan(start, len(text), "SYMPTOM", "suy tim", 0.9, 2),),
    )
    examples = build_training_examples(
        lattice,
        (GoldMention("doc", start, len(text), "suy tim", "DIAGNOSIS", "source-a"),),
        split="train",
        source_group="source-a",
    )
    example = examples[0]
    assert example.boundary_target.action == BOUNDARY_KEEP
    assert example.type_target.entity_type == "DIAGNOSIS"
    assert example.type_target.wrong_type_cost == 2.0
    assert example.type_target.diagnosis_symptom_pair


def test_learned_l4_constructs_bounded_offset_and_select_existing_targets() -> None:
    text = "suy tim nặng"
    lattice = build_span_lattice(
        "doc",
        text,
        neural_spans=(
            NeuralSpan(0, 3, "DIAGNOSIS", "suy", 0.7, 1),
            NeuralSpan(0, 7, "DIAGNOSIS", "suy tim", 0.9, 2),
        ),
    )
    gold = (GoldMention("doc", 0, 7, "suy tim", "DIAGNOSIS", "source-a"),)
    examples = build_training_examples(
        lattice,
        gold,
        split="validation",
        source_group="source-a",
        config=ResolverV2Config(max_boundary_delta=8),
    )
    short = next(example for example in examples if example.proposal_end == 3)
    exact = next(example for example in examples if example.proposal_end == 7)
    assert short.boundary_target.action == BOUNDARY_SELECT_EXISTING
    assert short.boundary_target.selected_proposal_id
    assert exact.boundary_target.action == BOUNDARY_KEEP

    offset_lattice = build_span_lattice(
        "doc2",
        text,
        neural_spans=(NeuralSpan(0, 3, "DIAGNOSIS", "suy", 0.7, 1),),
    )
    offset_examples = build_training_examples(
        offset_lattice,
        (GoldMention("doc2", 0, 7, "suy tim", "DIAGNOSIS", "source-a"),),
        split="train",
        source_group="source-a",
        config=ResolverV2Config(max_boundary_delta=8),
    )
    assert offset_examples[0].boundary_target.action == BOUNDARY_OFFSET
    assert offset_examples[0].boundary_target.end_delta == 4


def test_learned_l4_grouped_split_prevents_source_leakage() -> None:
    text = "suy tim"
    lattice = build_span_lattice(
        "doc",
        text,
        neural_spans=(NeuralSpan(0, 7, "DIAGNOSIS", "suy tim", 0.9, 2),),
    )
    gold = (GoldMention("doc", 0, 7, "suy tim", "DIAGNOSIS", "source-a"),)
    examples_a = build_training_examples(lattice, gold, split="train", source_group="source-a")
    examples_b = build_training_examples(lattice, gold, split="train", source_group="source-b")
    train, validation, split_id = grouped_train_validation_split(
        examples_a + examples_b,
        validation_fraction=0.5,
        seed=3,
    )
    assert split_id
    assert {example.source_group for example in train}.isdisjoint(
        {example.source_group for example in validation}
    )
    with pytest.raises(ResolverV2Error):
        build_training_examples(lattice, gold, split="internal_test", source_group="source-a")


def test_learned_l4_runtime_loads_read_only_json_checkpoint_and_resolves(tmp_path: Path) -> None:
    text = "suy tim"
    lattice = build_span_lattice(
        "doc",
        text,
        neural_spans=(NeuralSpan(0, 7, "DIAGNOSIS", "suy tim", 0.9, 2),),
    )
    checkpoint_path = tmp_path / "learned_l4_v2.json"
    digest = _write_checkpoint(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, expected_sha256=digest)
    result = resolve_lattice(lattice, checkpoint, ResolverV2Config(enabled=True))
    assert result.accepted()[0].text == "suy tim"
    assert result.decisions[0].boundary_action == BOUNDARY_KEEP
    assert result.decisions[0].wrong_type_risk < 0.35
    assert (
        result.determinism_hash()
        == resolve_lattice(lattice, checkpoint, ResolverV2Config(enabled=True)).determinism_hash()
    )


def test_checkpoint_backend_satisfies_the_learned_l4_protocol(tmp_path: Path) -> None:
    from mednorm_vi.resolution.learned_dispatch import (
        LEARNED_L4_DISPATCH_VERSION,
        CheckpointLearnedL4Backend,
        LearnedL4Backend,
    )

    checkpoint_path = tmp_path / "learned_l4_v2.json"
    digest = _write_checkpoint(checkpoint_path)
    config = ResolverV2Config(
        enabled=True,
        checkpoint_path=str(checkpoint_path),
        expected_checkpoint_sha256=digest,
    )
    backend = CheckpointLearnedL4Backend(
        checkpoint=validate_runtime_checkpoint(config),
        config=config,
    )
    assert isinstance(backend, LearnedL4Backend)
    assert backend.contract_version == LEARNED_L4_DISPATCH_VERSION
    assert backend.describe()["contract_version"] == LEARNED_L4_DISPATCH_VERSION


def test_learned_l4_abstains_on_high_wrong_type_risk(tmp_path: Path) -> None:
    text = "suy tim"
    lattice = build_span_lattice(
        "doc",
        text,
        neural_spans=(NeuralSpan(0, 7, "DIAGNOSIS", "suy tim", 0.9, 2),),
    )
    checkpoint_path = tmp_path / "learned_l4_v2.json"
    digest = _write_checkpoint(checkpoint_path, wrong_type_bias=4.0)
    checkpoint = load_checkpoint(checkpoint_path, expected_sha256=digest)
    result = resolve_lattice(lattice, checkpoint, ResolverV2Config(enabled=True))
    assert not result.accepted()
    assert result.decisions[0].status == "abstained"


def test_learned_l4_runtime_rejects_disabled_or_missing_checkpoint() -> None:
    with pytest.raises(ResolverV2UnavailableError):
        validate_runtime_checkpoint(ResolverV2Config(enabled=False))
    with pytest.raises(ResolverV2UnavailableError):
        validate_runtime_checkpoint(ResolverV2Config(enabled=True, checkpoint_path="/tmp/nope"))
