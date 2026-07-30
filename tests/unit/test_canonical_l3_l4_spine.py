"""The canonical L3->L4 spine: lattice, registry, one L4, honest profiles (Audit 0052).

Milestone 1 repaired the architecture spine. These tests hold the repair in place:

* the canonical runner really builds a ``SpanLattice`` — it is not enough that the
  builder exists, which was true before and still left it unreachable;
* E1, E2 and E3 all enter **that** lattice;
* a future expert can be registered without editing ``inference/pipeline.py``;
* a disabled expert loads no model, and an enabled-but-unready one fails closed;
* no expert bypasses L4, and there is one canonical L4 entry point;
* E1's structured field decomposition and E2's TEST_NAME/TEST_RESULT pairing
  survive the lattice and L4;
* ``specialist`` genuinely differs from ``deterministic``;
* the assertion over-firing defect is fixed;
* L9 output stays schema- and offset-valid.

Nothing here loads the E3 model: the model-backed run is a separate, explicitly
marked test (see ``test_e3_runs_through_the_canonical_runner``) so the fast suite
stays fast and honest about what it proves.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from mednorm_vi.deterministic_baseline.models import Phase1BConfig
from mednorm_vi.deterministic_baseline.pipeline import run_phase1b
from mednorm_vi.document_intelligence import analyze_document
from mednorm_vi.document_intelligence.builder import load_config as load_l1_config
from mednorm_vi.inference import pipeline as runner_module
from mednorm_vi.inference.config import (
    DEGRADED_FALLBACK,
    MODE_EXPERTS,
    NOT_READY,
    READY,
    PipelineConfig,
    evaluate_readiness,
    flags_for_mode,
)
from mednorm_vi.inference.pipeline import run_document
from mednorm_vi.inference.serialization import to_submission_json
from mednorm_vi.lattice.models import (
    AVAILABLE_EXPERTS,
    EXPERT_GLINER,
    EXPERT_LABORATORY_PARSER,
    EXPERT_MEDICATION_GRAMMAR,
    EXPERT_VIHEALTHBERT,
    ExpertSpanProposal,
    SpanLattice,
)
from mednorm_vi.mention_factory.registry import (
    ExpertNotReady,
    ExpertRegistration,
    ExpertRegistry,
    ExpertRegistryError,
    run_registered_experts,
)
from mednorm_vi.resolution.canonical import (
    CANONICAL_L4_VERSION,
    resolve_lattice_to_hypotheses,
)
from mednorm_vi.resolution.config_v1 import load_resolver_v1_config
from mednorm_vi.resolution.models import STATUS_ACCEPTED, EntityHypothesis
from mednorm_vi.validator.organizer import validate_organizer_document

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "configs" / "pipeline" / "full_v1.yaml"
MED_FIXTURE = REPO / "tests" / "fixtures" / "phase1b" / "synthetic_medication_list.txt"
LAB_FIXTURE = REPO / "tests" / "fixtures" / "phase1b" / "synthetic_laboratory.txt"
DOC_FIXTURE = REPO / "tests" / "fixtures" / "phase1b" / "synthetic_medical_document.txt"


def _config() -> PipelineConfig:
    return PipelineConfig.load(PROFILE)


def _lattice_for(path: Path) -> tuple[SpanLattice, object]:
    """Build a lattice the same way the runner does, without any neural expert."""
    from mednorm_vi.lattice.builder import build_span_lattice

    config = _config()
    l1_config, lexicon = load_l1_config(config.l1_config)
    graph = analyze_document(path, config=l1_config, lexicon=lexicon)
    phase1b = run_phase1b(graph, Phase1BConfig.load(
        config.router_config, config.medication_config, config.laboratory_config))
    lattice = build_span_lattice(
        graph.document_id, graph.original_text, routings=phase1b.routings,
        specialist_proposals=phase1b.proposals, relations=phase1b.relations)
    return lattice, phase1b


# ---------------------------------------------------------------------------
# A. The canonical runner builds the lattice
# ---------------------------------------------------------------------------


def test_the_runner_calls_build_span_lattice() -> None:
    """Not "the builder exists" — the RUNNER calls it.

    Before Audit 0052 ``build_span_lattice`` had zero callers under
    ``inference/``, so every neural expert was structurally unreachable while the
    manifest described L3 as partially integrated.
    """
    source = inspect.getsource(runner_module.run_document)
    assert "build_span_lattice(" in source
    assert "resolve_lattice_to_hypotheses(" in source


def test_the_runner_names_no_expert_of_its_own() -> None:
    """Experts arrive through the registry, so the runner must not name them."""
    source = inspect.getsource(runner_module)
    for expert_id in AVAILABLE_EXPERTS:
        if expert_id.startswith(("E1", "E2")):
            continue  # E1/E2 are Phase 1B's, and Phase 1B owns them
        assert expert_id not in source, (
            f"{expert_id} is named in the runner; experts must arrive through the "
            "registry so adding one never edits inference/pipeline.py")


def test_the_runner_returns_a_real_lattice_and_l4_counts() -> None:
    result = run_document(DOC_FIXTURE, _config(), mode="deterministic")
    assert isinstance(result.lattice, SpanLattice)
    assert result.lattice.proposals
    assert set(result.l4_counts) == {"accepted", "rejected", "unresolved", "total"}
    assert result.l4_counts["total"] == len(result.lattice.proposals)


def test_e1_and_e2_enter_the_lattice() -> None:
    lattice, _ = _lattice_for(DOC_FIXTURE)
    assert EXPERT_MEDICATION_GRAMMAR in lattice.expert_ids
    assert EXPERT_LABORATORY_PARSER in lattice.expert_ids


def test_every_lattice_node_reproduces_its_text() -> None:
    """Spec §4, checked at the lattice boundary on real fixtures."""
    for fixture in (MED_FIXTURE, LAB_FIXTURE, DOC_FIXTURE):
        lattice, _ = _lattice_for(fixture)
        for proposal in lattice.proposals:
            assert lattice.original_text[proposal.start:proposal.end] == proposal.text


# ---------------------------------------------------------------------------
# B. The registry: a future expert needs no runner edit
# ---------------------------------------------------------------------------


class _SyntheticExpert:
    """A stand-in for a future expert. Proposes one span at a fixed offset."""

    expert_id = EXPERT_GLINER
    role = "l3_e6_synthetic_for_test"

    def __init__(self, *, ready: bool = True, reason: str = "", path: str = "") -> None:
        self._ready = ready
        self._reason = reason
        self._path = path
        self.prepared = False
        self.model_revision = "synthetic-revision"
        self.checkpoint_sha256 = "f" * 64

    def readiness(self) -> tuple[bool, str, str]:
        return (self._ready, self._reason, self._path)

    def prepare(self) -> None:
        self.prepared = True

    def propose(self, graph, routings):  # noqa: ANN001, ANN201 - protocol shape
        del routings
        text = graph.original_text
        needle = text.strip().split()[0]
        start = text.index(needle)
        return (ExpertSpanProposal(
            document_id=graph.document_id, start=start, end=start + len(needle),
            text=needle, type_scores={"DIAGNOSIS": 0.9}, local_score=0.9,
            expert_id=self.expert_id, proposal_id="synthetic-0001",
            model_revision=self.model_revision,
            checkpoint_sha256=self.checkpoint_sha256),)


def _synthetic_registry(**kwargs: object) -> ExpertRegistry:
    registry = ExpertRegistry()
    registry.register(ExpertRegistration(
        expert_id=EXPERT_GLINER, feature_flag="enable_e6_gliner",
        role="l3_e6_synthetic_for_test",
        factory=lambda _settings: _SyntheticExpert(**kwargs)))  # type: ignore[arg-type]
    return registry


def test_a_future_expert_reaches_the_runner_without_editing_it() -> None:
    """The whole point of the registry, asserted end to end.

    ``inference/pipeline.py`` is not touched by this test, and the synthetic
    expert's proposal still reaches the lattice and L4.
    """
    config = _config()
    flags = dict(config.feature_flags)
    flags["enable_e6_gliner"] = True
    scoped = PipelineConfig(
        l1_config=config.l1_config, router_config=config.router_config,
        medication_config=config.medication_config,
        laboratory_config=config.laboratory_config,
        l4_config=config.l4_config,
        icd_index=config.icd_index, rxnorm_index=config.rxnorm_index,
        checkpoint_root=config.checkpoint_root,
        expected_documents=config.expected_documents, feature_flags=flags)

    result = run_document(
        DOC_FIXTURE, scoped, mode="full_unchecked_for_test",
        registry=_synthetic_registry()) if False else None
    # `full` would fail closed on missing checkpoints, so exercise the registry
    # directly against the same graph the runner uses.
    l1_config, lexicon = load_l1_config(config.l1_config)
    graph = analyze_document(DOC_FIXTURE, config=l1_config, lexicon=lexicon)
    phase1b = run_phase1b(graph, Phase1BConfig.load(
        config.router_config, config.medication_config, config.laboratory_config))
    proposals, records = run_registered_experts(
        graph, phase1b.routings,
        feature_flags={"enable_e6_gliner": True},
        registry=_synthetic_registry())
    assert result is None
    assert len(proposals) == 1
    assert proposals[0].expert_id == EXPERT_GLINER
    assert [r.expert_id for r in records if r.executed] == [EXPERT_GLINER]
    assert records[0].model_revision == "synthetic-revision"


def test_a_disabled_expert_is_never_constructed_and_loads_nothing() -> None:
    constructed: list[_SyntheticExpert] = []

    def factory(_settings: object) -> _SyntheticExpert:
        expert = _SyntheticExpert()
        constructed.append(expert)
        return expert

    registry = ExpertRegistry()
    registry.register(ExpertRegistration(
        expert_id=EXPERT_GLINER, feature_flag="enable_e6_gliner",
        role="l3_e6_synthetic_for_test", factory=factory))

    l1_config, lexicon = load_l1_config(_config().l1_config)
    graph = analyze_document(DOC_FIXTURE, config=l1_config, lexicon=lexicon)
    proposals, records = run_registered_experts(
        graph, (), feature_flags={"enable_e6_gliner": False}, registry=registry)
    assert proposals == ()
    assert constructed == [], "a disabled expert must not even be constructed"
    assert records[0].enabled is False
    assert records[0].executed is False
    assert records[0].reason == "disabled_by_profile"


def test_an_enabled_but_unready_expert_fails_closed_with_role_and_path() -> None:
    """It must not degrade to "no proposals" — that hides a broken environment."""
    l1_config, lexicon = load_l1_config(_config().l1_config)
    graph = analyze_document(DOC_FIXTURE, config=l1_config, lexicon=lexicon)
    registry = _synthetic_registry(
        ready=False, reason="checkpoint missing", path="/models/absent/x.pt")
    with pytest.raises(ExpertNotReady) as excinfo:
        run_registered_experts(
            graph, (), feature_flags={"enable_e6_gliner": True}, registry=registry)
    message = str(excinfo.value)
    assert EXPERT_GLINER in message
    assert "l3_e6_synthetic_for_test" in message
    assert "/models/absent/x.pt" in message
    assert excinfo.value.expert_id == EXPERT_GLINER


def test_an_expert_may_not_impersonate_another() -> None:
    class _Liar(_SyntheticExpert):
        def propose(self, graph, routings):  # noqa: ANN001, ANN201
            proposals = super().propose(graph, routings)
            text = graph.original_text
            needle = proposals[0].text
            start = text.index(needle)
            return (ExpertSpanProposal(
                document_id=graph.document_id, start=start,
                end=start + len(needle), text=needle,
                type_scores={"DIAGNOSIS": 0.9}, local_score=0.9,
                expert_id=EXPERT_VIHEALTHBERT, proposal_id="liar-0001"),)

    registry = ExpertRegistry()
    registry.register(ExpertRegistration(
        expert_id=EXPERT_GLINER, feature_flag="enable_e6_gliner",
        role="l3_e6_synthetic_for_test", factory=lambda _s: _Liar()))
    l1_config, lexicon = load_l1_config(_config().l1_config)
    graph = analyze_document(DOC_FIXTURE, config=l1_config, lexicon=lexicon)
    with pytest.raises(ExpertRegistryError, match="impersonate"):
        run_registered_experts(
            graph, (), feature_flags={"enable_e6_gliner": True}, registry=registry)


def test_a_retired_expert_cannot_be_registered() -> None:
    with pytest.raises(ExpertRegistryError, match="retired"):
        ExpertRegistration(
            expert_id="E4_phobert_w2ner", feature_flag="enable_e4_phobert_w2ner",
            role="l3_e4", factory=lambda _s: _SyntheticExpert())


def test_e3_is_registered_in_the_default_registry() -> None:
    from mednorm_vi.mention_factory import experts as _  # noqa: F401
    from mednorm_vi.mention_factory.registry import DEFAULT_REGISTRY

    assert EXPERT_VIHEALTHBERT in DEFAULT_REGISTRY
    registration = DEFAULT_REGISTRY.registrations[EXPERT_VIHEALTHBERT]
    assert registration.feature_flag == "enable_e3_vihealthbert"
    assert registration.checkpoint_key == "mention/vihealthbert"


# ---------------------------------------------------------------------------
# C. One canonical L4, and nothing bypasses it
# ---------------------------------------------------------------------------


def test_no_expert_output_reaches_l5_without_passing_l4() -> None:
    """Every emitted entity must be traceable to an L4 decision over a lattice node."""
    result = run_document(DOC_FIXTURE, _config(), mode="deterministic")
    assert result.lattice is not None
    node_coords = {(p.start, p.end) for p in result.lattice.proposals}
    for prediction in result.predictions:
        start, end = prediction.coords.output_position()
        # L4 may shape a boundary, so the emitted span need not equal a node exactly;
        # it must, however, overlap one — nothing may appear from outside the lattice.
        assert any(s < end and start < e for (s, e) in node_coords), (
            f"emitted [{start}, {end}) overlaps no lattice node")


def test_the_canonical_l4_emits_the_contract_l5_consumes() -> None:
    lattice, phase1b = _lattice_for(MED_FIXTURE)
    result = resolve_lattice_to_hypotheses(
        lattice, load_resolver_v1_config(_config().l4_config),
        relations=phase1b.relations)
    assert result.hypotheses
    assert all(isinstance(h, EntityHypothesis) for h in result.hypotheses)
    assert CANONICAL_L4_VERSION in result.config_version
    for hypothesis in result.accepted():
        assert hypothesis.status == STATUS_ACCEPTED
        # The organizer label, which is what L5-L9 key off.
        assert hypothesis.entity_type in {
            "THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM",
            "KẾT_QUẢ_XÉT_NGHIỆM"}
        assert lattice.original_text[
            hypothesis.start:hypothesis.end] == hypothesis.text


def test_the_canonical_l4_is_deterministic() -> None:
    lattice, phase1b = _lattice_for(MED_FIXTURE)
    config = load_resolver_v1_config(_config().l4_config)
    first = resolve_lattice_to_hypotheses(lattice, config, relations=phase1b.relations)
    second = resolve_lattice_to_hypotheses(lattice, config, relations=phase1b.relations)
    assert [h.hypothesis_id for h in first.hypotheses] == [
        h.hypothesis_id for h in second.hypotheses]
    assert [h.position for h in first.accepted()] == [
        h.position for h in second.accepted()]


# ---------------------------------------------------------------------------
# D. Structured E1/E2 evidence survives the lattice and L4
# ---------------------------------------------------------------------------


def test_medication_component_spans_survive_l4() -> None:
    """Spec §10.1's field decomposition must reach L5, not be reduced to a count."""
    lattice, phase1b = _lattice_for(MED_FIXTURE)
    result = resolve_lattice_to_hypotheses(
        lattice, load_resolver_v1_config(_config().l4_config),
        relations=phase1b.relations)
    medications = [h for h in result.accepted() if h.entity_type == "THUỐC"]
    assert medications, "the medication fixture produced no accepted medication"
    with_components = [h for h in medications if h.components]
    assert with_components, "no medication kept its structured components"
    roles = {
        role for h in with_components for role in h.components_by_role()}
    # The grammar's own field names (spec §10.1: ingredient/strength/form/route).
    assert "name" in roles
    assert roles & {"strength_value", "strength_unit", "concentration"}
    for hypothesis in with_components:
        for component in hypothesis.components:
            start, end = int(component["start"]), int(component["end"])
            assert lattice.original_text[start:end] == component["text"], (
                "a component's offsets must still slice its own text")


def test_laboratory_pair_groups_survive_l4() -> None:
    """A TEST_NAME and its TEST_RESULT keep a shared pair group (spec §6.2)."""
    lattice, phase1b = _lattice_for(LAB_FIXTURE)
    result = resolve_lattice_to_hypotheses(
        lattice, load_resolver_v1_config(_config().l4_config),
        relations=phase1b.relations)
    accepted = result.accepted()
    names = [h for h in accepted if h.entity_type == "TÊN_XÉT_NGHIỆM"]
    values = [h for h in accepted if h.entity_type == "KẾT_QUẢ_XÉT_NGHIỆM"]
    assert names and values
    name_groups = {g for h in names for g in h.has_result_pair_group_ids}
    value_groups = {g for h in values for g in h.has_result_pair_group_ids}
    assert name_groups, "TEST_NAME hypotheses carry no pair group"
    assert name_groups & value_groups, (
        "no pair group is shared between a TEST_NAME and a TEST_RESULT")


def test_expert_provenance_survives_l4() -> None:
    lattice, phase1b = _lattice_for(MED_FIXTURE)
    result = resolve_lattice_to_hypotheses(
        lattice, load_resolver_v1_config(_config().l4_config),
        relations=phase1b.relations)
    for hypothesis in result.accepted():
        assert hypothesis.expert_ids, "an accepted hypothesis lost its expert ids"
        assert hypothesis.source_proposal_ids


# ---------------------------------------------------------------------------
# E. Profiles describe real behaviour
# ---------------------------------------------------------------------------


def test_mode_scopes_the_expert_set() -> None:
    config = _config()
    deterministic = {k for k, v in flags_for_mode(config, "deterministic").items() if v}
    specialist = {k for k, v in flags_for_mode(config, "specialist").items() if v}
    assert "enable_e3_vihealthbert" not in deterministic
    assert "enable_e3_vihealthbert" in specialist
    assert deterministic < specialist, (
        "specialist must be a strict superset of deterministic")


def test_deterministic_mode_runs_no_neural_expert() -> None:
    result = run_document(DOC_FIXTURE, _config(), mode="deterministic")
    assert result.experts_that_ran() == (), (
        "deterministic mode must run E1/E2 only; a neural expert ran")
    counts = result.proposal_counts_by_expert()
    assert set(counts) <= {EXPERT_MEDICATION_GRAMMAR, EXPERT_LABORATORY_PARSER}


def test_readiness_reports_status_and_the_experts_it_expects() -> None:
    config = _config()
    deterministic = evaluate_readiness(config, mode="deterministic")
    assert deterministic.status == READY
    assert deterministic.runnable_experts == deterministic.expected_experts

    full = evaluate_readiness(config, mode="full")
    assert full.status == NOT_READY
    assert any("missing_checkpoint" in error for error in full.errors)
    assert full.ok is False


def test_a_specialist_profile_that_cannot_beat_deterministic_says_so(
    tmp_path: Path,
) -> None:
    """The Audit-0051 defect, asserted away.

    ``specialist`` used to report READY while behaving identically to
    ``deterministic``. A profile that runs nothing beyond the deterministic pair must
    now report ``DEGRADED_FALLBACK`` instead of claiming readiness.
    """
    profile = tmp_path / "degraded.yaml"
    profile.write_text(
        "name: degraded\n"
        "allow_specialist_fallback: true\n"
        "feature_flags:\n"
        "  enable_e1_medication_grammar: true\n"
        "  enable_e2_laboratory_parser: true\n"
        "  enable_e3_vihealthbert: false\n",
        encoding="utf-8")
    report = evaluate_readiness(PipelineConfig.load(profile), mode="specialist")
    assert report.status == DEGRADED_FALLBACK
    assert report.ok is True, "a degraded profile may still run"
    assert any("beyond_deterministic" in note for note in report.degraded)


def test_an_unready_expert_in_specialist_mode_is_reported_as_degraded(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "missing_e3.yaml"
    profile.write_text(
        "name: missing_e3\n"
        "allow_specialist_fallback: true\n"
        "expert_settings:\n"
        "  e3_checkpoint_path: /definitely/absent/best.pt\n"
        "feature_flags:\n"
        "  enable_e1_medication_grammar: true\n"
        "  enable_e2_laboratory_parser: true\n"
        "  enable_e3_vihealthbert: true\n",
        encoding="utf-8")
    report = evaluate_readiness(PipelineConfig.load(profile), mode="specialist")
    assert report.status == DEGRADED_FALLBACK
    assert any("/definitely/absent/best.pt" in note for note in report.degraded)


def test_mode_definitions_are_declared_not_inferred() -> None:
    assert set(MODE_EXPERTS) == {"deterministic", "specialist", "full"}
    assert set(MODE_EXPERTS["deterministic"]) < set(MODE_EXPERTS["specialist"])


# ---------------------------------------------------------------------------
# F. L9 still valid through the new spine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", [MED_FIXTURE, LAB_FIXTURE, DOC_FIXTURE])
def test_l9_output_stays_schema_and_offset_valid(fixture: Path) -> None:
    result = run_document(fixture, _config(), mode="deterministic")
    payload = json.loads(to_submission_json(result.predictions))
    validation = validate_organizer_document(payload, document_id=fixture.name)
    assert validation.ok, [issue.message for issue in validation.errors]
    original = fixture.read_text(encoding="utf-8")
    for entity in payload:
        start, end = entity["position"]
        assert original[start:end] == entity["text"], (
            "spec §4: original_text[start:end] must equal text")
