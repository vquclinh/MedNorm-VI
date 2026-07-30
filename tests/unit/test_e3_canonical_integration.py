"""E3 runs through the canonical runner, with real weights (Audit 0052).

E3 has had a trained, validated checkpoint since Audit 0031 and could not run in
production until Audit 0052, because the canonical runner had no path for a neural
expert. These tests prove the path exists and that the real model traverses it.

Everything that loads the model is skipped unless the verified checkpoint and its
dependencies are present, so the suite stays green on a machine without them — and
the skip reason names exactly what is missing rather than passing quietly.

Nothing here trains, downloads or writes: the checkpoint is opened read-only, its
digest is verified before and after inference, and no weight is modified.
"""

from __future__ import annotations

import hashlib
from importlib.util import find_spec
from pathlib import Path

import pytest

from mednorm_vi.inference.config import PipelineConfig, evaluate_readiness
from mednorm_vi.inference.pipeline import run_document
from mednorm_vi.lattice.models import EXPERT_VIHEALTHBERT
from mednorm_vi.mention_factory.experts.e3_vihealthbert import (
    E3_CHECKPOINT_SHA256,
    E3_PINNED_MODEL_REVISION,
    E3VietHealthBertExpert,
)

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "configs" / "pipeline" / "full_v1.yaml"
FIXTURE = REPO / "tests" / "fixtures" / "phase1b" / "synthetic_medical_document.txt"
CHECKPOINT = REPO / "checkpoint" / "s1_mention_full_training_v1" / "best.pt"
EXPECTED_SIZE = 1_615_513_303


def _missing() -> str:
    if not CHECKPOINT.is_file():
        return f"E3 checkpoint absent: {CHECKPOINT}"
    for dependency in ("torch", "transformers"):
        try:
            if find_spec(dependency) is None:
                return f"{dependency} not installed"
        except (ImportError, ValueError):
            return f"{dependency} not importable"
    return ""


requires_e3 = pytest.mark.skipif(bool(_missing()), reason=_missing() or "ok")


# ---------------------------------------------------------------------------
# Contract-only: no model is loaded
# ---------------------------------------------------------------------------


def test_constructing_the_expert_loads_nothing() -> None:
    expert = E3VietHealthBertExpert()
    assert expert.loaded is False
    assert expert.expert_id == EXPERT_VIHEALTHBERT


def test_readiness_does_not_load_a_model() -> None:
    expert = E3VietHealthBertExpert()
    expert.readiness()
    assert expert.loaded is False, "readiness must never construct the model"


def test_a_missing_checkpoint_is_reported_not_raised() -> None:
    expert = E3VietHealthBertExpert(checkpoint_path="/definitely/absent/best.pt")
    ready, reason, path = expert.readiness()
    assert ready is False
    assert "does not exist" in reason
    assert path == "/definitely/absent/best.pt"


def test_a_vncorenlp_directory_without_a_jar_is_refused(tmp_path: Path) -> None:
    """E3 was trained on RDRSegmenter output; silent whitespace fallback is wrong."""
    expert = E3VietHealthBertExpert(vncorenlp_dir=str(tmp_path))
    ready, reason, _path = expert.readiness()
    assert ready is False
    assert reason == "e3_vncorenlp_not_ready:missing_jar"


def test_the_pinned_identity_matches_the_validated_artifact() -> None:
    """These two values tie the runtime to Audits 0031/0032's validated artifact."""
    assert E3_CHECKPOINT_SHA256 == (
        "a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c"
    )
    assert E3_PINNED_MODEL_REVISION == "f89e80b461e86f9cfc1c84019bd819830c24b6c5"


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="checkpoint absent")
def test_the_checkpoint_on_disk_is_the_verified_artifact() -> None:
    assert CHECKPOINT.stat().st_size == EXPECTED_SIZE
    digest = hashlib.sha256()
    with CHECKPOINT.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    assert digest.hexdigest() == E3_CHECKPOINT_SHA256


# ---------------------------------------------------------------------------
# With real weights
# ---------------------------------------------------------------------------


@requires_e3
def test_specialist_mode_is_ready_and_names_e3() -> None:
    report = evaluate_readiness(PipelineConfig.load(PROFILE), mode="specialist")
    assert "enable_e3_vihealthbert" in report.expected_experts
    assert "enable_e3_vihealthbert" in report.runnable_experts


@requires_e3
def test_e3_runs_through_the_canonical_runner() -> None:
    """The milestone's central claim, proven with real weights.

    Not "the adapter exists" and not "the checkpoint validates" — the model runs
    inside ``run_document`` and its proposals reach the lattice.
    """
    result = run_document(FIXTURE, PipelineConfig.load(PROFILE), mode="specialist")
    assert EXPERT_VIHEALTHBERT in result.experts_that_ran()
    assert result.lattice is not None
    assert EXPERT_VIHEALTHBERT in result.lattice.expert_ids

    record = next(r for r in result.expert_records if r.expert_id == EXPERT_VIHEALTHBERT)
    assert record.executed is True
    assert record.proposal_count > 0
    # Provenance the ledger and any future audit needs.
    assert record.model_revision == E3_PINNED_MODEL_REVISION
    assert record.checkpoint_sha256 == E3_CHECKPOINT_SHA256


@requires_e3
def test_e3_proposals_satisfy_the_offset_invariant() -> None:
    result = run_document(FIXTURE, PipelineConfig.load(PROFILE), mode="specialist")
    text = FIXTURE.read_text(encoding="utf-8")
    for proposal in result.lattice.proposals:
        for source in proposal.sources:
            if source.expert_id != EXPERT_VIHEALTHBERT:
                continue
            assert text[proposal.start : proposal.end] == proposal.text


@requires_e3
def test_specialist_differs_from_deterministic_with_real_weights() -> None:
    config = PipelineConfig.load(PROFILE)
    deterministic = run_document(FIXTURE, config, mode="deterministic")
    specialist = run_document(FIXTURE, config, mode="specialist")
    assert deterministic.experts_that_ran() == ()
    assert EXPERT_VIHEALTHBERT in specialist.experts_that_ran()
    assert len(specialist.lattice.proposals) > len(deterministic.lattice.proposals), (
        "E3 contributed no lattice node, so the two modes are indistinguishable"
    )


@requires_e3
def test_e3_emits_no_final_entity_and_cannot_bypass_l4() -> None:
    """Spec §6: no expert emits a final entity."""
    from mednorm_vi.lattice.models import ExpertSpanProposal

    expert = E3VietHealthBertExpert()
    ready, reason, _ = expert.readiness()
    assert ready, reason
    expert.prepare()

    from mednorm_vi.document_intelligence import analyze_document
    from mednorm_vi.document_intelligence.builder import load_config

    config = PipelineConfig.load(PROFILE)
    l1_config, lexicon = load_config(config.l1_config)
    graph = analyze_document(FIXTURE, config=l1_config, lexicon=lexicon)
    proposals = expert.propose(graph, ())
    assert proposals
    assert all(isinstance(p, ExpertSpanProposal) for p in proposals), "E3 must emit proposals only"
    # The checkpoint must be untouched by inference.
    expert.verify_checkpoint_unchanged()


@requires_e3
def test_the_checkpoint_is_unchanged_after_a_full_run() -> None:
    before = CHECKPOINT.stat().st_mtime_ns
    run_document(FIXTURE, PipelineConfig.load(PROFILE), mode="specialist")
    digest = hashlib.sha256()
    with CHECKPOINT.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    assert digest.hexdigest() == E3_CHECKPOINT_SHA256
    assert CHECKPOINT.stat().st_mtime_ns == before, "inference modified the checkpoint"
