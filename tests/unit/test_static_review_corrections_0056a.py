"""Audit 0056a — the independent-review corrections, each held by a test.

Every test here corresponds to a defect an independent review reported against
Audit 0055 and this milestone fixed. They are grouped by the finding they close, and
each names the specific untruth it prevents from returning.

Nothing in this file loads a model, reads the real 1.6 GB E3 checkpoint, executes a
notebook, builds a container or runs corpus inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Finding A — readiness may not claim READY for something that cannot execute
# ---------------------------------------------------------------------------


def _base_profile(tmp_path: Path, **overrides: Any) -> Path:
    payload: dict[str, Any] = {
        "name": "test_profile",
        "l1_config": "configs/document_intelligence/base.yaml",
        "router_config": "configs/case_router/base.yaml",
        "medication_config": "configs/medication/grammar_v1.yaml",
        "laboratory_config": "configs/laboratory/parser_v1.yaml",
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "expected_documents": 2,
    }
    payload.update(overrides)
    target = tmp_path / "profile.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return target


def test_an_enabled_but_unregistered_expert_makes_the_profile_not_ready(
    tmp_path: Path,
) -> None:
    """E5/E6/E7 were enabled-able and invisible to readiness (finding A).

    `expected` was built from MODE_EXPERTS["full"] — E1/E2/E3 only — so the one
    function that asks whether an expert can run was never consulted for them.
    """
    from mednorm_vi.inference.config import NOT_READY, PipelineConfig, evaluate_readiness

    config = PipelineConfig.load(
        _base_profile(
            tmp_path,
            feature_flags={
                "enable_e3_vihealthbert": False,
                "enable_e5_xlmr_mrc": True,
                "enable_e6_gliner": True,
                "enable_e7_qwen_proposer": True,
            },
        )
    )
    report = evaluate_readiness(config, mode="full")
    assert report.status == NOT_READY
    # Every enabled expert is now EXPECTED, which is what makes it checkable.
    for flag in ("enable_e5_xlmr_mrc", "enable_e6_gliner", "enable_e7_qwen_proposer"):
        assert flag in report.expected_experts
        assert flag not in report.runnable_experts
    joined = " ".join(report.errors)
    assert "E5 has no registered expert" in joined
    assert "E6 has no registered expert" in joined
    assert "E7 has no registered expert" in joined


def test_an_empty_checkpoint_directory_does_not_satisfy_readiness(
    tmp_path: Path,
) -> None:
    """`Path.exists()` is True for an empty directory — three mkdirs bought READY."""
    from mednorm_vi.inference.config import NOT_READY, PipelineConfig, evaluate_readiness

    root = tmp_path / "checkpoints"
    for role in ("mention/xlmr_mrc", "mention/gliner", "mention/qwen3_1_7b_proposer"):
        (root / role).mkdir(parents=True)

    config = PipelineConfig.load(
        _base_profile(
            tmp_path,
            feature_flags={
                "enable_e3_vihealthbert": False,
                "enable_e5_xlmr_mrc": True,
                "enable_e6_gliner": True,
                "enable_e7_qwen_proposer": True,
            },
        )
    )
    report = evaluate_readiness(config, mode="full")
    assert report.status == NOT_READY, "an empty directory must never satisfy checkpoint readiness"


def test_a_fake_but_nonempty_artifact_still_fails_on_the_missing_implementation(
    tmp_path: Path,
) -> None:
    """Even a plausible-looking artifact cannot make an unregistered expert runnable.

    This is the sharper half of finding A: the reviewer's "fake checkpoint
    directories may satisfy path-only readiness checks". Filling the directory
    defeats an artifact check but must not defeat the implementation check.
    """
    from mednorm_vi.inference.config import NOT_READY, PipelineConfig, evaluate_readiness

    root = tmp_path / "checkpoints" / "mention" / "gliner"
    root.mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"not really a model")

    config = PipelineConfig.load(
        _base_profile(
            tmp_path, feature_flags={"enable_e3_vihealthbert": False, "enable_e6_gliner": True}
        )
    )
    report = evaluate_readiness(config, mode="full")
    assert report.status == NOT_READY
    assert any("E6 has no registered expert" in error for error in report.errors)


def test_a_registered_expert_without_its_asset_is_not_ready(tmp_path: Path) -> None:
    """E3 IS registered, so its blocker comes from the adapter, not the spec."""
    from mednorm_vi.inference.config import NOT_READY, PipelineConfig, evaluate_readiness

    config = PipelineConfig.load(
        _base_profile(
            tmp_path,
            feature_flags={"enable_e3_vihealthbert": True},
            specialist_requires_checkpoints=["mention/vihealthbert"],
            allow_specialist_fallback=False,
            expert_settings={"e3_checkpoint_path": str(tmp_path / "absent" / "best.pt")},
        )
    )
    report = evaluate_readiness(config, mode="specialist")
    assert report.status == NOT_READY
    assert any("does not exist" in error for error in report.errors)


def test_a_disabled_expert_is_never_constructed(tmp_path: Path) -> None:
    """A disabled expert must cost no memory and touch no checkpoint."""
    from mednorm_vi.lattice.models import EXPERT_VIHEALTHBERT
    from mednorm_vi.mention_factory.registry import (
        ExpertRegistration,
        ExpertRegistry,
        run_registered_experts,
    )

    constructed: list[str] = []

    def factory(_settings: Any) -> Any:
        constructed.append("built")
        raise AssertionError("a disabled expert must not be constructed")

    registry = ExpertRegistry()
    registry.register(
        ExpertRegistration(
            expert_id=EXPERT_VIHEALTHBERT,
            feature_flag="enable_e3_vihealthbert",
            role="test",
            factory=factory,
        )
    )
    proposals, records = run_registered_experts(
        graph=object(),
        routings=(),
        feature_flags={"enable_e3_vihealthbert": False},
        registry=registry,
    )
    assert proposals == ()
    assert constructed == []
    assert [record.reason for record in records] == ["disabled_by_profile"]


def test_the_expert_specification_and_the_registry_cannot_drift() -> None:
    """The four views of an expert must agree, or readiness says so by name.

    Before Audit 0056a the profile flags, MODE_EXPERTS, the registry and the
    required-asset lists each held a partial view and could disagree silently.
    """
    from mednorm_vi.mention_factory.expert_spec import (
        EXPERT_SPECS,
        IMPLEMENTATION_ABSENT,
        IMPLEMENTATION_DETERMINISTIC,
        IMPLEMENTATION_REGISTERED,
    )
    from mednorm_vi.mention_factory.registry import DEFAULT_REGISTRY

    registered = {r.feature_flag for r in DEFAULT_REGISTRY.ordered()}
    for spec in EXPERT_SPECS:
        if spec.implementation == IMPLEMENTATION_REGISTERED:
            assert spec.feature_flag in registered, (
                f"{spec.expert_id} claims REGISTERED but nothing registers its flag"
            )
        elif spec.implementation == IMPLEMENTATION_ABSENT:
            assert spec.feature_flag not in registered, (
                f"{spec.expert_id} is declared absent but IS registered; a "
                "registration must never be added merely to make readiness pass"
            )
            assert spec.blocker, "an absent expert must name its blocker"
        else:
            assert spec.implementation == IMPLEMENTATION_DETERMINISTIC


def test_every_registered_expert_is_declared_by_the_specification() -> None:
    """The drift check in the other direction: no registration without a spec."""
    from mednorm_vi.mention_factory.expert_spec import EXPERT_SPEC_BY_FLAG
    from mednorm_vi.mention_factory.registry import DEFAULT_REGISTRY

    for registration in DEFAULT_REGISTRY.ordered():
        assert registration.feature_flag in EXPERT_SPEC_BY_FLAG


# ---------------------------------------------------------------------------
# Finding G — a declared requirement must be read
# ---------------------------------------------------------------------------


def test_specialist_checkpoint_requirements_are_enforced(tmp_path: Path) -> None:
    """`specialist_requires_checkpoints` was parsed and never read (finding G)."""
    from mednorm_vi.inference.config import NOT_READY, PipelineConfig, evaluate_readiness

    config = PipelineConfig.load(
        _base_profile(
            tmp_path,
            feature_flags={"enable_e3_vihealthbert": True},
            specialist_requires_checkpoints=["retrieval/icd_dense"],
        )
    )
    report = evaluate_readiness(config, mode="specialist")
    assert report.status == NOT_READY
    assert any(
        error.startswith("missing_required_checkpoint:retrieval/icd_dense")
        for error in report.errors
    )


def test_the_shipped_profile_requires_only_real_checkpoint_backed_assets() -> None:
    """`assertion/assertion_hydra` is deterministic logic, not a checkpoint.

    Requiring it would make `specialist` permanently NOT_READY for an artifact the
    architecture never plans to produce — a false negative is no more honest than a
    false positive.
    """
    from mednorm_vi.inference.config import PipelineConfig

    config = PipelineConfig.load(REPO / "configs" / "pipeline" / "full_v1.yaml")
    assert config.specialist_requires_checkpoints == ("mention/vihealthbert",)


def test_specialist_stays_ready_with_the_real_e3_asset() -> None:
    """The shipped profile must remain usable. Reads no weights: readiness only."""
    from mednorm_vi.inference.config import READY, PipelineConfig, evaluate_readiness

    checkpoint = REPO / "checkpoint" / "s1_mention_full_training_v1" / "best.pt"
    if not checkpoint.is_file():
        pytest.skip("the real E3 checkpoint is not present in this environment")
    config = PipelineConfig.load(REPO / "configs" / "pipeline" / "full_v1.yaml")
    report = evaluate_readiness(config, mode="specialist")
    assert report.status == READY, report.errors
    assert "enable_e3_vihealthbert" in report.runnable_experts


# ---------------------------------------------------------------------------
# Finding B — both L4 flags have real semantics; the learned route fails closed
# ---------------------------------------------------------------------------


def test_no_l4_route_selected_is_refused_at_load(tmp_path: Path) -> None:
    """L4 is not optional: L5-L9 consume EntityHypothesis and nothing else makes it."""
    from mednorm_vi.inference.config import PipelineConfig

    profile = _base_profile(
        tmp_path, feature_flags={"enable_l4_deterministic_v1": False, "enable_l4_learned_v2": False}
    )
    with pytest.raises(ValueError) as raised:
        PipelineConfig.load(profile)
    assert "no L4 route is selected" in str(raised.value)


def test_both_l4_routes_selected_is_refused_at_load(tmp_path: Path) -> None:
    from mednorm_vi.inference.config import PipelineConfig

    profile = _base_profile(
        tmp_path, feature_flags={"enable_l4_deterministic_v1": True, "enable_l4_learned_v2": True}
    )
    with pytest.raises(ValueError) as raised:
        PipelineConfig.load(profile)
    assert "ambiguous" in str(raised.value)


def test_the_shipped_profile_selects_the_deterministic_route() -> None:
    """The flag was `false` while the deterministic resolver ran regardless."""
    from mednorm_vi.inference.config import (
        L4_ROUTE_DETERMINISTIC_V1,
        PipelineConfig,
        select_l4_route,
    )

    config = PipelineConfig.load(REPO / "configs" / "pipeline" / "full_v1.yaml")
    assert select_l4_route(config.feature_flags) == L4_ROUTE_DETERMINISTIC_V1
    assert config.feature_flags["enable_l4_deterministic_v1"] is True


def test_the_deterministic_route_resolves_no_learned_backend(tmp_path: Path) -> None:
    from mednorm_vi.inference.config import PipelineConfig
    from mednorm_vi.inference.pipeline import resolve_l4_backend

    config = PipelineConfig.load(_base_profile(tmp_path))
    assert resolve_l4_backend(config) is None, (
        "the deterministic route must not construct a learned backend"
    )


def test_enabling_learned_l4_without_a_checkpoint_fails_closed(tmp_path: Path) -> None:
    """The central no-silent-fallback guarantee (finding B)."""
    from mednorm_vi.inference.config import PipelineConfig
    from mednorm_vi.inference.pipeline import resolve_l4_backend
    from mednorm_vi.resolution.learned_dispatch import LearnedL4Unavailable

    config = PipelineConfig.load(
        _base_profile(
            tmp_path,
            feature_flags={"enable_l4_deterministic_v1": False, "enable_l4_learned_v2": True},
        )
    )
    with pytest.raises(LearnedL4Unavailable) as raised:
        resolve_l4_backend(config)
    message = str(raised.value)
    assert "NOT substituted" in message, "the refusal must state that it did not fall back"


def test_enabling_learned_l4_without_a_checkpoint_is_a_readiness_blocker(
    tmp_path: Path,
) -> None:
    """An operator learns the route is unusable before a run, not on document 1."""
    from mednorm_vi.inference.config import NOT_READY, PipelineConfig, evaluate_readiness

    config = PipelineConfig.load(
        _base_profile(
            tmp_path,
            feature_flags={"enable_l4_deterministic_v1": False, "enable_l4_learned_v2": True},
        )
    )
    report = evaluate_readiness(config, mode="deterministic")
    assert report.status == NOT_READY
    assert any(error.startswith("l4_learned_not_ready:") for error in report.errors)


class _StubLearnedBackend:
    """An injected learned backend. Loads nothing; returns a fixed result."""

    contract_version = "test-learned-l4"

    def __init__(self, result: Any = None) -> None:
        self.result = result
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        return {"contract_version": self.contract_version}

    def resolve(self, lattice: Any, *, relations: Any = ()) -> Any:
        from mednorm_vi.resolution.models import ResolutionResult

        self.calls += 1
        if self.result is not None:
            return self.result
        return ResolutionResult(
            document_id=lattice.document_id, hypotheses=(), config_version=self.contract_version
        )


def test_an_injected_learned_backend_is_reached_by_the_canonical_l4() -> None:
    """Dispatch is verified with a stub, never by loading a model."""
    from mednorm_vi.lattice.models import SpanLattice
    from mednorm_vi.resolution.canonical import resolve_lattice_to_hypotheses
    from mednorm_vi.resolution.config_v1 import load_resolver_v1_config

    backend = _StubLearnedBackend()
    lattice = SpanLattice(document_id="d1", original_text="aspirin", proposals=())
    result = resolve_lattice_to_hypotheses(
        lattice,
        load_resolver_v1_config(
            str(REPO / "configs" / "resolution" / "boundary_type_resolver_v1.yaml")
        ),
        learned_backend=backend,
    )
    assert backend.calls == 1, "the learned route must reach the backend"
    assert result.config_version == backend.contract_version


def test_the_canonical_l4_returns_one_contract_on_both_routes() -> None:
    """One canonical EntityHypothesis output contract, whichever route ran."""
    from mednorm_vi.lattice.models import SpanLattice
    from mednorm_vi.resolution.canonical import resolve_lattice_to_hypotheses
    from mednorm_vi.resolution.config_v1 import load_resolver_v1_config
    from mednorm_vi.resolution.models import ResolutionResult

    config = load_resolver_v1_config(
        str(REPO / "configs" / "resolution" / "boundary_type_resolver_v1.yaml")
    )
    lattice = SpanLattice(document_id="d1", original_text="aspirin", proposals=())
    deterministic = resolve_lattice_to_hypotheses(lattice, config)
    learned = resolve_lattice_to_hypotheses(lattice, config, learned_backend=_StubLearnedBackend())
    assert isinstance(deterministic, ResolutionResult)
    assert isinstance(learned, ResolutionResult)


def test_learned_output_that_breaks_the_offset_invariant_is_rejected() -> None:
    """Learned output is untrusted input; a bad coordinate rejects the document."""
    from mednorm_vi.lattice.models import SpanLattice
    from mednorm_vi.resolution.learned_dispatch import (
        LearnedL4OutputRejected,
        _adapt_result,
    )
    from mednorm_vi.resolution.learned_v2 import ResolverV2Decision, ResolverV2Result
    from mednorm_vi.schemas.hypotheses import TypedHypothesis
    from mednorm_vi.schemas.spans import Span, SpanCoordinates

    lattice = SpanLattice(document_id="d1", original_text="uống aspirin", proposals=())
    bad = TypedHypothesis(
        hypothesis_id="h1",
        text="paracetamol",  # not what [5, 12) slices out of the source
        coords=SpanCoordinates(absolute=Span(5, 12)),
        type_distribution={"MEDICATION": 1.0},
    )
    inner = ResolverV2Result(
        document_id="d1",
        hypotheses=(bad,),
        decisions=(
            ResolverV2Decision(
                hypothesis_id="h1",
                proposal_id="p1",
                status="accepted",
                boundary_action="KEEP",
                original_start=5,
                original_end=12,
                start=5,
                end=12,
                entity_type="MEDICATION",
                calibrated_score=0.9,
                wrong_type_risk=0.0,
                reason="accepted",
            ),
        ),
        config_sha256="x",
        checkpoint_sha256="y",
    )
    with pytest.raises(LearnedL4OutputRejected):
        _adapt_result(lattice, inner, backend_version="test")


def test_learned_output_outside_the_document_is_rejected() -> None:
    from mednorm_vi.lattice.models import SpanLattice
    from mednorm_vi.resolution.learned_dispatch import (
        LearnedL4OutputRejected,
        _adapt_result,
    )
    from mednorm_vi.resolution.learned_v2 import ResolverV2Decision, ResolverV2Result
    from mednorm_vi.schemas.hypotheses import TypedHypothesis
    from mednorm_vi.schemas.spans import Span, SpanCoordinates

    lattice = SpanLattice(document_id="d1", original_text="aspirin", proposals=())
    out_of_range = TypedHypothesis(
        hypothesis_id="h1",
        text="aspirin",
        coords=SpanCoordinates(absolute=Span(0, 999)),
        type_distribution={"MEDICATION": 1.0},
    )
    inner = ResolverV2Result(
        document_id="d1",
        hypotheses=(out_of_range,),
        decisions=(
            ResolverV2Decision(
                hypothesis_id="h1",
                proposal_id="p1",
                status="accepted",
                boundary_action="KEEP",
                original_start=0,
                original_end=7,
                start=0,
                end=999,
                entity_type="MEDICATION",
                calibrated_score=0.9,
                wrong_type_risk=0.0,
                reason="accepted",
            ),
        ),
        config_sha256="x",
        checkpoint_sha256="y",
    )
    with pytest.raises(LearnedL4OutputRejected):
        _adapt_result(lattice, inner, backend_version="test")


# ---------------------------------------------------------------------------
# Findings D and E — the final L9 gate over the serialized payload
# ---------------------------------------------------------------------------

_SOURCE = "Chẩn đoán: viêm phổi. Thuốc: aspirin"
_DIAGNOSIS_START = _SOURCE.index("viêm phổi")
_DIAGNOSIS_END = _DIAGNOSIS_START + len("viêm phổi")


class _StubSnapshot:
    """A locked snapshot stand-in. Imports no linker and no model."""

    def __init__(self, index_type: str, codes: set[str]) -> None:
        self._index_type = index_type
        self._codes = codes

    @property
    def index_type(self) -> str:
        return self._index_type

    @property
    def source_snapshot_id(self) -> str:
        return f"{self._index_type}-test-snapshot"

    def exists(self, concept_id: str) -> bool:
        return concept_id in self._codes


def _snapshots() -> Any:
    from mednorm_vi.validator.kb_membership import LockedSnapshots

    return LockedSnapshots(
        icd10=_StubSnapshot("icd10_vi", {"J18", "J189"}),
        rxnorm=_StubSnapshot("rxnorm", {"1191"}),
    )


def _entity(**overrides: Any) -> dict[str, Any]:
    entity: dict[str, Any] = {
        "text": "viêm phổi",
        "type": "CHẨN_ĐOÁN",
        "candidates": ["J18"],
        "assertions": [],
        "position": [_DIAGNOSIS_START, _DIAGNOSIS_END],
    }
    entity.update(overrides)
    return entity


def _document(entities: list[dict[str, Any]], offered: dict[int, tuple[str, ...]]) -> Any:
    from mednorm_vi.validator.final_gate import FinalDocument

    return FinalDocument(
        document_id="1",
        filename="1.json",
        original_text=_SOURCE,
        payload=json.dumps(entities, ensure_ascii=False),
        offered_codes_by_index=offered,
    )


def test_a_valid_final_entity_passes_the_gate() -> None:
    from mednorm_vi.validator.final_gate import validate_final_document

    result = validate_final_document(_document([_entity()], {0: ("J18", "J189")}), _snapshots())
    assert result.ok, [issue.code for issue in result.errors]


def test_an_offset_text_mismatch_fails_the_gate() -> None:
    """Appendix A: every entity satisfies original_text[start:end] == text."""
    from mednorm_vi.validator.final_gate import validate_final_document

    result = validate_final_document(
        _document([_entity(text="viêm phổi nặng")], {0: ("J18",)}), _snapshots()
    )
    assert not result.ok
    assert "final.text_offset_mismatch" in {issue.code for issue in result.errors}


def test_a_span_outside_the_document_fails_the_gate() -> None:
    from mednorm_vi.validator.final_gate import validate_final_document

    result = validate_final_document(
        _document([_entity(position=[0, 10_000])], {0: ("J18",)}), _snapshots()
    )
    assert not result.ok
    assert "final.span_outside_document" in {issue.code for issue in result.errors}


def test_a_snapshot_valid_but_unoffered_code_fails_the_gate() -> None:
    """Spec P7: existing in the KB is not the same as having been retrieved."""
    from mednorm_vi.validator.final_gate import validate_final_document

    result = validate_final_document(
        # J189 is in the snapshot, but only J18 was offered for this mention.
        _document([_entity(candidates=["J189"])], {0: ("J18",)}),
        _snapshots(),
    )
    assert not result.ok
    assert "kb.candidate_not_offered" in {issue.code for issue in result.errors}


def test_an_invented_code_fails_the_gate() -> None:
    from mednorm_vi.validator.final_gate import validate_final_document

    result = validate_final_document(
        _document([_entity(candidates=["Z999"])], {0: ("J18",)}), _snapshots()
    )
    assert not result.ok
    codes = {issue.code for issue in result.errors}
    assert "kb.candidate_not_in_snapshot" in codes


def test_a_wrong_ontology_code_fails_the_gate() -> None:
    """An RxCUI on a DIAGNOSIS is a category error, not a near miss."""
    from mednorm_vi.validator.final_gate import validate_final_document

    result = validate_final_document(
        _document([_entity(candidates=["1191"])], {0: ("1191",)}), _snapshots()
    )
    assert not result.ok


def test_a_duplicate_candidate_fails_the_gate() -> None:
    from mednorm_vi.validator.final_gate import validate_final_document

    result = validate_final_document(
        _document([_entity(candidates=["J18", "J18"])], {0: ("J18",)}), _snapshots()
    )
    assert not result.ok
    assert "final.duplicate_candidate" in {issue.code for issue in result.errors}


def test_an_entity_emitting_candidates_with_no_recorded_offered_set_fails() -> None:
    """A missing offered set would silently disable P7 for that entity."""
    from mednorm_vi.validator.final_gate import validate_final_document

    result = validate_final_document(_document([_entity()], {}), _snapshots())
    assert not result.ok
    assert "final.offered_set_missing" in {issue.code for issue in result.errors}


def test_out_of_order_entities_fail_the_gate() -> None:
    """Appendix A's deterministic ordering, checkable on a single payload."""
    from mednorm_vi.validator.final_gate import validate_final_document

    first = _entity(position=[_DIAGNOSIS_START, _DIAGNOSIS_END])
    second = _entity(text="Chẩn đoán", type="TRIỆU_CHỨNG", position=[0, 9])
    second.pop("candidates")
    result = validate_final_document(_document([first, second], {0: ("J18",)}), _snapshots())
    assert not result.ok
    assert "final.nondeterministic_order" in {issue.code for issue in result.errors}


def test_an_unsafe_output_filename_fails_the_gate() -> None:
    from mednorm_vi.validator.final_gate import validate_safe_filename

    assert validate_safe_filename("1.json", "1").ok
    assert not validate_safe_filename("../1.json", "1").ok
    assert not validate_safe_filename("sub/1.json", "1").ok
    assert not validate_safe_filename("output.json", "1").ok


def test_gate_final_documents_raises_and_names_every_violation() -> None:
    from mednorm_vi.validator.final_gate import FinalValidationError, gate_final_documents

    with pytest.raises(FinalValidationError) as raised:
        gate_final_documents(
            [_document([_entity(candidates=["Z999"])], {0: ("Z999",)})], _snapshots()
        )
    assert "1.json" in str(raised.value)


# ---------------------------------------------------------------------------
# Finding E — the gate is ON the canonical packaging path
# ---------------------------------------------------------------------------


def _deterministic_profile(tmp_path: Path) -> Any:
    from mednorm_vi.inference.config import PipelineConfig

    return PipelineConfig.load(
        _base_profile(tmp_path, feature_flags={"enable_e3_vihealthbert": False})
    )


def test_a_post_l8_candidate_injection_stops_packaging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer's scenario: a code introduced after L8 must not be packaged.

    The injection is applied to the SERIALIZED payload, which is exactly where a
    future reranker or LLM selector could introduce one, and exactly what the
    pre-0056a gate could not see.
    """
    from mednorm_vi.inference import pipeline as pipeline_module
    from mednorm_vi.validator.final_gate import FinalValidationError

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text("Thuốc: aspirin 81mg\n", encoding="utf-8")

    original = pipeline_module.to_submission_json

    def injecting(predictions: Any) -> str:
        payload = json.loads(original(predictions))
        payload.append(
            {
                "text": "aspirin",
                "type": "THUỐC",
                "candidates": ["999999"],
                "assertions": [],
                "position": [7, 14],
            }
        )
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr(pipeline_module, "to_submission_json", injecting)

    with pytest.raises(FinalValidationError):
        pipeline_module.run_input_dir(
            input_dir,
            output_zip=tmp_path / "output.zip",
            config=_deterministic_profile(tmp_path),
            mode="deterministic",
        )


def test_a_stopped_run_leaves_no_output_directory_and_no_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §16 forbids silent repair: a violating run must package nothing."""
    from mednorm_vi.inference import pipeline as pipeline_module
    from mednorm_vi.validator.final_gate import FinalValidationError

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text("Thuốc: aspirin 81mg\n", encoding="utf-8")

    def corrupting(predictions: Any) -> str:
        # An offset that cannot slice out of the source document.
        return json.dumps(
            [
                {
                    "text": "khong-ton-tai",
                    "type": "THUỐC",
                    "candidates": [],
                    "assertions": [],
                    "position": [0, 13],
                }
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr(pipeline_module, "to_submission_json", corrupting)

    output_zip = tmp_path / "output.zip"
    with pytest.raises(FinalValidationError):
        pipeline_module.run_input_dir(
            input_dir,
            output_zip=output_zip,
            config=_deterministic_profile(tmp_path),
            mode="deterministic",
        )
    assert not output_zip.exists(), "no output.zip may survive a failed final gate"
    assert not (tmp_path / "output").exists(), "no output/ may survive"


def test_the_canonical_runner_carries_source_text_and_offered_sets(
    tmp_path: Path,
) -> None:
    """The two inputs the pre-0056a gate never received."""
    from mednorm_vi.inference.pipeline import run_document

    document = tmp_path / "1.txt"
    document.write_text("Thuốc: aspirin 81mg\n", encoding="utf-8")
    result = run_document(document, _deterministic_profile(tmp_path), mode="deterministic")
    assert result.original_text == "Thuốc: aspirin 81mg\n"
    assert isinstance(result.offered_codes_by_index, dict)


# ---------------------------------------------------------------------------
# Finding I — checkpoint deserialization policy
# ---------------------------------------------------------------------------


def test_only_the_trusted_legacy_digest_reaches_weights_only_false(
    tmp_path: Path,
) -> None:
    from mednorm_vi.mention_factory.neural.checkpoint_loading import (
        LOAD_MODE_TRUSTED_LEGACY_PICKLE,
        TRUSTED_LEGACY_E3_SHA256,
        assert_load_is_permitted,
    )

    legacy = tmp_path / "best.pt"
    legacy.write_bytes(b"synthetic")
    policy = assert_load_is_permitted(legacy, verified_sha256=TRUSTED_LEGACY_E3_SHA256)
    assert policy.mode == LOAD_MODE_TRUSTED_LEGACY_PICKLE
    assert policy.weights_only is False
    assert policy.is_trusted_legacy


def test_an_unrecognised_checkpoint_is_forced_to_safe_loading(tmp_path: Path) -> None:
    """The legacy allowance must not become the house style."""
    from mednorm_vi.mention_factory.neural.checkpoint_loading import (
        LOAD_MODE_WEIGHTS_ONLY,
        assert_load_is_permitted,
    )

    future = tmp_path / "future.pt"
    future.write_bytes(b"synthetic")
    policy = assert_load_is_permitted(future, verified_sha256="a" * 64)
    assert policy.mode == LOAD_MODE_WEIGHTS_ONLY
    assert policy.weights_only is True


def test_an_unverified_checkpoint_cannot_be_loaded_at_all(tmp_path: Path) -> None:
    """No digest, no load — the ordering guarantee made structural."""
    from mednorm_vi.mention_factory.neural.checkpoint_loading import (
        CheckpointTrustError,
        assert_load_is_permitted,
    )

    target = tmp_path / "unverified.pt"
    target.write_bytes(b"synthetic")
    with pytest.raises(CheckpointTrustError) as raised:
        assert_load_is_permitted(target, verified_sha256="")
    assert "weights_only=False" in str(raised.value)


def test_a_digest_mismatch_is_refused_before_any_load(tmp_path: Path) -> None:
    from mednorm_vi.mention_factory.neural.checkpoint_loading import (
        CheckpointTrustError,
        assert_load_is_permitted,
    )

    target = tmp_path / "substituted.pt"
    target.write_bytes(b"synthetic")
    with pytest.raises(CheckpointTrustError):
        assert_load_is_permitted(target, verified_sha256="b" * 64, expected_sha256="a" * 64)


def test_safetensors_is_safe_by_format(tmp_path: Path) -> None:
    from mednorm_vi.mention_factory.neural.checkpoint_loading import (
        LOAD_MODE_SAFETENSORS,
        assert_load_is_permitted,
    )

    target = tmp_path / "model.safetensors"
    target.write_bytes(b"synthetic")
    policy = assert_load_is_permitted(target, verified_sha256="")
    assert policy.mode == LOAD_MODE_SAFETENSORS
    assert policy.weights_only is True


def test_the_e3_runtime_asks_the_policy_rather_than_hardcoding_the_flag() -> None:
    """A source-level assertion: no bare `weights_only=False` on the E3 path."""
    source = (REPO / "src" / "mednorm_vi" / "mention_factory" / "neural" / "runtime.py").read_text(
        encoding="utf-8"
    )
    assert "weights_only=policy.weights_only" in source
    # Comments explaining the policy may name the flag; live code may not set it.
    code = "\n".join(
        line for line in source.splitlines() if line.strip() and not line.strip().startswith("#")
    )
    assert "weights_only=False" not in code, (
        "the E3 loader must obtain its load mode from assert_load_is_permitted"
    )


# ---------------------------------------------------------------------------
# Finding J — output-directory safety
# ---------------------------------------------------------------------------


def test_writing_refuses_a_directory_that_is_not_named_output(tmp_path: Path) -> None:
    from mednorm_vi.inference.packaging import UnsafeOutputDirectory, write_output_directory

    target = tmp_path / "important"
    target.mkdir()
    (target / "keep.txt").write_text("do not delete", encoding="utf-8")
    with pytest.raises(UnsafeOutputDirectory):
        write_output_directory({"1.json": "[]"}, target)
    assert (target / "keep.txt").is_file(), "nothing may be deleted before the check"


def test_writing_refuses_an_output_directory_holding_unrelated_files(
    tmp_path: Path,
) -> None:
    from mednorm_vi.inference.packaging import UnsafeOutputDirectory, write_output_directory

    target = tmp_path / "output"
    target.mkdir()
    (target / "notes.txt").write_text("someone else's data", encoding="utf-8")
    with pytest.raises(UnsafeOutputDirectory):
        write_output_directory({"1.json": "[]"}, target)
    assert (target / "notes.txt").is_file()


def test_writing_refuses_a_symlinked_destination(tmp_path: Path) -> None:
    from mednorm_vi.inference.packaging import (
        UnsafeOutputDirectory,
        write_output_directory,
    )

    real = tmp_path / "real"
    real.mkdir()
    (real / "keep.txt").write_text("target", encoding="utf-8")
    link = tmp_path / "output"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(UnsafeOutputDirectory):
        write_output_directory({"1.json": "[]"}, link)
    assert (real / "keep.txt").is_file()


def test_writing_refuses_output_reached_through_a_symlinked_parent(
    tmp_path: Path,
) -> None:
    from mednorm_vi.inference.packaging import (
        UnsafeOutputDirectory,
        write_output_directory,
    )

    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    sentinel = real_parent / "keep.txt"
    sentinel.write_text("target", encoding="utf-8")
    symlink_parent = tmp_path / "symlink_parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(UnsafeOutputDirectory):
        write_output_directory({"1.json": "[]"}, symlink_parent / "output")
    assert sentinel.is_file()
    assert not (real_parent / "output").exists()


def test_writing_refuses_home_and_filesystem_root() -> None:
    from mednorm_vi.inference.packaging import (
        UnsafeOutputDirectory,
        assert_safe_output_directory,
    )

    with pytest.raises(UnsafeOutputDirectory):
        assert_safe_output_directory(Path("/"))
    with pytest.raises(UnsafeOutputDirectory):
        assert_safe_output_directory(Path.home())


def test_writing_refuses_the_repository_root() -> None:
    from mednorm_vi.inference.packaging import (
        UnsafeOutputDirectory,
        assert_safe_output_directory,
    )

    with pytest.raises(UnsafeOutputDirectory):
        assert_safe_output_directory(REPO)


def test_writing_accepts_an_absent_or_previous_output_directory(tmp_path: Path) -> None:
    """Normal inference behaviour is preserved."""
    from mednorm_vi.inference.packaging import write_output_directory

    target = tmp_path / "output"
    written = write_output_directory({"1.json": "[]"}, target)
    assert (written / "1.json").is_file()
    # A second run replaces a directory this workflow itself wrote.
    again = write_output_directory({"1.json": "[]", "2.json": "[]"}, target)
    assert sorted(p.name for p in again.iterdir()) == ["1.json", "2.json"]


# ---------------------------------------------------------------------------
# Finding C — E4 may not appear in executable notebook source
# ---------------------------------------------------------------------------

_E4_ACTIVE_TOKENS: tuple[str, ...] = (
    "enable_e4_phobert_w2ner",
    "validate_e4_artifact",
    "E4_FULL_ARTIFACT",
    "mention/phobert_w2ner",
    "e4_phobert_w2ner",
    "phobert-base",
)


def test_no_executable_notebook_cell_references_active_e4() -> None:
    """Static only: the notebook JSON is parsed, never executed."""
    offenders: list[str] = []
    for notebook in sorted((REPO / "notebooks").glob("*.ipynb")):
        document = json.loads(notebook.read_text(encoding="utf-8"))
        for index, cell in enumerate(document.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            # A comment recording the retirement is allowed; live code is not.
            code_lines = [
                line
                for line in source.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            code = "\n".join(code_lines)
            for token in _E4_ACTIVE_TOKENS:
                if token in code:
                    offenders.append(f"{notebook.name} cell {index}: {token}")
    assert not offenders, (
        "E4 is RETIRED_FROM_ACTIVE_ARCHITECTURE and must not appear in executable "
        f"notebook code: {offenders}"
    )


def test_the_s0_notebook_is_an_explicit_design_draft() -> None:
    """S0 must not silently retarget to another model (owner decision, 0056a)."""
    document = json.loads(
        (REPO / "notebooks" / "MedNorm_S0_DomainAdaptation.ipynb").read_text(encoding="utf-8")
    )
    cells = document.get("cells", [])
    header = "".join(cells[0].get("source", []))
    assert "DESIGN_DRAFT" in header
    assert "no s0 target model is selected" in header.lower()

    code = "\n".join(
        "".join(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "code"
    )
    assert "S0_TARGET_MODEL" in code
    # The unset target must fail closed rather than defaulting to a model.
    assert "raise SystemExit" in code


def _executable_code(notebook: Path) -> str:
    """Executable lines of every code cell, with comments stripped.

    Comments recording the retirement are allowed and expected; only live code is
    checked, which is what "executable cells must not treat E4 as supported" means.
    """
    document = json.loads(notebook.read_text(encoding="utf-8"))
    lines: list[str] = []
    for cell in document.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in "".join(cell.get("source", [])).splitlines():
            if line.strip() and not line.strip().startswith("#"):
                lines.append(line)
    return "\n".join(lines)


def test_the_phase2_ablation_notebook_declares_only_valid_arms() -> None:
    code = _executable_code(REPO / "notebooks" / "MedNorm_Phase2_Validation_Ablation.ipynb")
    assert "validate_e5_artifact" in code
    assert "validate_l4_artifact" in code
    assert "validate_e4_artifact" not in code


def test_the_e3_alignment_module_is_untouched_infrastructure() -> None:
    """`phobert_alignment` is E3's, not E4's, and must not be removed as a remnant.

    ViHealthBERT-Word declares `tokenizer_class = PhobertTokenizer`, which has no
    fast implementation, so this module reconstructs the offset mapping E3's
    supervision depends on.
    """
    module = REPO / "src" / "mednorm_vi" / "training" / "phobert_alignment.py"
    assert module.is_file()
    assert "vihealthbert" in module.read_text(encoding="utf-8").lower()


# ---------------------------------------------------------------------------
# Finding K — the legacy adapter layer is gone, with no shim
# ---------------------------------------------------------------------------


def test_the_legacy_neural_adapter_module_is_absent() -> None:
    assert not (REPO / "src" / "mednorm_vi" / "mention_factory" / "adapters.py").exists()


def test_the_legacy_neural_adapter_module_is_unimportable() -> None:
    """No alias, no compatibility shim: importing it must fail loudly."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mednorm_vi.mention_factory.adapters")


def test_the_future_expert_slot_configs_agree_with_the_canonical_specification() -> None:
    """E5/E6/E7 YAML slots are KEPT, and held to the specification.

    Audit 0056a inspected these and found them read by **no source module** — they
    are typed-slot declarations for future models, not runtime configuration. They
    are retained deliberately (they record the expert id, status, unpinned revision
    and planned parameter count a future download must satisfy), but a declaration
    nothing reads is exactly the defect class this audit closed elsewhere, so this
    test consumes them: a slot file that names an expert the canonical
    specification does not declare, or that claims to be enabled while the expert
    is unimplemented, fails here.
    """
    from mednorm_vi.mention_factory.expert_spec import (
        EXPERT_SPEC_BY_ID,
        IMPLEMENTATION_ABSENT,
    )

    slots = {
        "configs/mention_factory/xlmr_mrc_ner_v1.yaml": "E5_xlmr_mrc_ner",
        "configs/mention_factory/gliner_v1.yaml": "E6_gliner_open_type",
        "configs/mention_factory/qwen_proposer_v1.yaml": "E7_qwen3_1_7b_proposer",
    }
    for relative, expected_id in slots.items():
        document = yaml.safe_load((REPO / relative).read_text(encoding="utf-8"))
        assert document["expert_id"] == expected_id, relative
        spec = EXPERT_SPEC_BY_ID[expected_id]
        assert spec.implementation == IMPLEMENTATION_ABSENT, relative
        assert document["enabled"] is False, (
            f"{relative} claims enabled while {expected_id} has no implementation"
        )


def test_no_tracked_source_references_the_deleted_adapter_module() -> None:
    roots = (REPO / "src", REPO / "tests", REPO / "notebooks", REPO / "configs")
    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in (".py", ".ipynb", ".yaml", ".yml"):
                continue
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "mention_factory.adapters" in text and "deleted" not in text:
                offenders.append(str(path.relative_to(REPO)))
    assert not offenders, offenders
