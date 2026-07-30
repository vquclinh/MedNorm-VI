"""Route gating, the honest L8 decoder, L9 KB membership, and the Docker contract.

Audit 0053. Each section holds one of the milestone's measured claims in place.

The route-gating section is worth reading for the diagnosis it encodes: E1 and E2 were
**already** route-gated in code (E2 refuses any node without C2; E1 requires C1/C5 or a
C3 node with strong medication evidence), so the Audit-0052 false positives did not
come from an expert running outside its route. They came from the **router** attaching
C2 to narrative prose: `key_value_row 0.30 + table_row 0.25 + numeric_present 0.30` =
0.85, past the 0.50 threshold, with no laboratory content at all. The fix is
positive-evidence gating in the router's own config, and these tests pin both halves —
that prose no longer routes to C2, and that a unit-less lab row still does.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mednorm_vi.case_router.signals import (
    _NUMERIC_KEY_VALUE,
    CaseSpec,
    SignalSpec,
    load_router_config,
)
from mednorm_vi.deterministic_baseline.models import Phase1BConfig
from mednorm_vi.deterministic_baseline.pipeline import run_phase1b
from mednorm_vi.document_intelligence import analyze_document
from mednorm_vi.document_intelligence.builder import load_config as load_l1_config
from mednorm_vi.inference.config import PipelineConfig
from mednorm_vi.inference.pipeline import run_document, run_input_dir
from mednorm_vi.inference.serialization import to_submission_json
from mednorm_vi.kb.indexing.retrieval import load_index
from mednorm_vi.lattice.models import (
    EXPERT_LABORATORY_PARSER,
    EXPERT_MEDICATION_GRAMMAR,
)
from mednorm_vi.linking.models import LinkedCandidate
from mednorm_vi.mention_factory.route_gate import (
    EXPERT_ROUTES,
    build_route_gate_diagnostics,
)
from mednorm_vi.metric_decoder import (
    CANDIDATE_SAFETY_BOUND,
    CalibratedDecoderUnavailable,
    decode_entities,
    decode_expected_jaccard_calibrated,
    decoder_status,
)
from mednorm_vi.metric_decoder.decoder import (
    DROP_DUPLICATE,
    DROP_SAFETY_BOUND,
    KEEP_EXACT,
    _select_candidates,
)
from mednorm_vi.validator.kb_membership import (
    LockedSnapshots,
    validate_document_candidates,
    validate_entity_candidates,
)
from mednorm_vi.validator.organizer import validate_organizer_document

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "configs" / "pipeline" / "full_v1.yaml"
FIX = REPO / "tests" / "fixtures" / "phase1b"


def _config() -> PipelineConfig:
    return PipelineConfig.load(PROFILE)


def _phase1b(path: Path):
    config = _config()
    l1_config, lexicon = load_l1_config(config.l1_config)
    graph = analyze_document(path, config=l1_config, lexicon=lexicon)
    return graph, run_phase1b(
        graph,
        Phase1BConfig.load(
            config.router_config, config.medication_config, config.laboratory_config
        ),
    )


# ---------------------------------------------------------------------------
# A. Route gating: the router now requires positive evidence
# ---------------------------------------------------------------------------


def test_c2_declares_required_evidence_in_config() -> None:
    config = load_router_config(_config().router_config)
    c2 = config.case_spec("C2")
    assert c2 is not None
    assert c2.required_evidence, "C2 must declare positive laboratory evidence"
    # `numeric_value` fires on any numeral and was present in every measured false
    # positive; including it would leave the defect untouched.
    assert "numeric_value" not in c2.required_evidence


def test_required_evidence_is_enforced_not_merely_documented() -> None:
    """The weights alone permit the defect: 0.30 + 0.25 = 0.55 > 0.50 threshold."""
    spec = CaseSpec(
        case="CX",
        name="x",
        activated_specialists=(),
        signals=(SignalSpec("a", "soft", 0.3, "structural.is_key_value_row"),),
        required_evidence=("lab_unit",),
    )
    assert spec.evidence_satisfied(frozenset({"lab_unit"})) is True
    assert spec.evidence_satisfied(frozenset({"a", "table_row"})) is False
    # No declared requirement means score alone decides.
    open_spec = CaseSpec("CY", "y", (), (), ())
    assert open_spec.evidence_satisfied(frozenset()) is True


@pytest.mark.parametrize(
    "text",
    [
        "WBC:14.43",
        "NEUT%: 76.4 %",
        "PLT: 250",  # unit-less result — spec §14.2 requires this to work
        "HGB: 120 g/L H",
        "Glucose: 7.8 mmol/L (3.9-6.4)",
    ],
)
def test_numeric_key_value_fires_on_real_lab_rows(text: str) -> None:
    assert _NUMERIC_KEY_VALUE.search(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "ung thư thận giai đoạn sớm ( 1,2 ) : phẫu thuật : tùy vào tính chất khối u",
        "bụng chướng , gõ rất trong và trung tiện nhiều lần ( ở người bình thường",
        "Tiền sử gia đình: bố tăng huyết áp",
        "Chẩn đoán: viêm phổi thùy dưới phải",
    ],
)
def test_numeric_key_value_does_not_fire_on_prose(text: str) -> None:
    """The value after the colon is text, not a number."""
    assert _NUMERIC_KEY_VALUE.search(text) is None


def test_laboratory_fixture_recall_is_preserved() -> None:
    """Route gating must not suppress a valid laboratory block."""
    _graph, phase1b = _phase1b(FIX / "synthetic_laboratory.txt")
    lab = [p for p in phase1b.proposals if p.source_specialist == "laboratory"]
    assert len(lab) >= 18, f"laboratory recall regressed to {len(lab)}"
    assert any("C2" in r.route_tags for r in phase1b.routings)


def test_medication_fixture_recall_is_preserved() -> None:
    _graph, phase1b = _phase1b(FIX / "synthetic_medication_list.txt")
    med = [p for p in phase1b.proposals if p.source_specialist == "medication"]
    assert len(med) >= 19, f"medication recall regressed to {len(med)}"
    assert any("C1" in r.route_tags for r in phase1b.routings)


def test_mixed_medication_and_laboratory_document_keeps_both() -> None:
    """A multi-label document must activate both experts, not choose one."""
    _graph, phase1b = _phase1b(FIX / "synthetic_medical_document.txt")
    med = [p for p in phase1b.proposals if p.source_specialist == "medication"]
    lab = [p for p in phase1b.proposals if p.source_specialist == "laboratory"]
    assert len(med) == 13, f"medication count changed: {len(med)}"
    assert len(lab) == 14, f"laboratory count changed: {len(lab)}"


def test_multi_label_routing_survives_the_gate() -> None:
    """No forced single-route classification (spec §5)."""
    _graph, phase1b = _phase1b(FIX / "synthetic_medical_document.txt")
    multi = [r for r in phase1b.routings if len(r.route_tags) > 1]
    assert multi, "no node carries more than one route; routing collapsed to one label"


def test_hard_negative_fixture_still_yields_nothing() -> None:
    _graph, phase1b = _phase1b(FIX / "synthetic_hard_negatives.txt")
    assert phase1b.proposals == ()


def test_a_suppressed_route_is_recorded_never_silent() -> None:
    """A route considered and withheld must be auditable."""
    config = _config()
    l1_config, lexicon = load_l1_config(config.l1_config)
    scratch = REPO / "tests" / "fixtures" / "phase1b" / "synthetic_medical_document.txt"
    graph = analyze_document(scratch, config=l1_config, lexicon=lexicon)
    phase1b = run_phase1b(
        graph,
        Phase1BConfig.load(
            config.router_config, config.medication_config, config.laboratory_config
        ),
    )
    assert any(r.gate_reasons for r in phase1b.routings), (
        "no node recorded a gate reason; suppression is invisible"
    )


# ---------------------------------------------------------------------------
# B. Route-gate diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_report_all_five_required_views() -> None:
    result = run_document(FIX / "synthetic_medical_document.txt", _config(), mode="deterministic")
    assert result.route_gate is not None
    payload = result.route_gate.as_dict()
    for key in (
        "eligible_nodes_by_expert",
        "skipped_nodes_by_expert",
        "proposals_by_expert",
        "active_routes_by_node",
        "route_gate_reasons",
    ):
        assert key in payload, f"missing required diagnostic {key}"
    assert payload["eligible_nodes_by_expert"][EXPERT_MEDICATION_GRAMMAR] > 0
    assert payload["eligible_nodes_by_expert"][EXPERT_LABORATORY_PARSER] > 0
    assert payload["skipped_nodes_by_expert"][EXPERT_LABORATORY_PARSER] > 0


def test_diagnostics_carry_no_clinical_text() -> None:
    result = run_document(FIX / "synthetic_medical_document.txt", _config(), mode="deterministic")
    payload = json.dumps(result.route_gate.as_dict(), ensure_ascii=False)
    original = (FIX / "synthetic_medical_document.txt").read_text(encoding="utf-8")
    for line in original.splitlines():
        stripped = line.strip()
        if len(stripped) >= 12:
            assert stripped not in payload, "diagnostics leaked document text"
    assert result.route_gate.as_dict()["contains_clinical_text"] is False


def test_expert_routes_are_declared_for_both_deterministic_experts() -> None:
    assert set(EXPERT_ROUTES) == {EXPERT_MEDICATION_GRAMMAR, EXPERT_LABORATORY_PARSER}
    assert "C2" in EXPERT_ROUTES[EXPERT_LABORATORY_PARSER]
    assert "C1" in EXPERT_ROUTES[EXPERT_MEDICATION_GRAMMAR]


def test_diagnostics_are_empty_but_valid_with_no_routings() -> None:
    diagnostics = build_route_gate_diagnostics("doc", ())
    payload = diagnostics.as_dict()
    assert payload["active_routes_by_node"] == {}
    assert payload["route_gate_reasons"] == []


# ---------------------------------------------------------------------------
# C. L8: no longer pretends, no longer a fixed top-10
# ---------------------------------------------------------------------------


def test_the_misleading_name_is_gone() -> None:
    import mednorm_vi.metric_decoder as decoder_package

    assert not hasattr(decoder_package, "decode_expected_jaccard"), (
        "the old name asserted expected-Jaccard decoding the body never did"
    )
    assert hasattr(decoder_package, "decode_entities")


def test_decoder_status_states_plainly_what_it_is_not() -> None:
    status = decoder_status()
    assert status["performs_expected_jaccard_decoding"] is False
    assert status["uses_calibrated_probabilities"] is False
    assert status["fixed_top_k_used"] is False
    assert status["calibrated_decoder_available"] is False


def test_the_calibrated_decoder_fails_closed() -> None:
    with pytest.raises(CalibratedDecoderUnavailable, match="calibrated probabilities"):
        decode_expected_jaccard_calibrated()


def _candidate(code: str, score: float, channel: str) -> LinkedCandidate:
    return LinkedCandidate(code=code, score=score, channels=(channel,), snapshot_id="snap")


def test_an_exact_alias_hit_is_emitted_alone() -> None:
    """Spec §9.2: the exact alias wins outright; weaker evidence beside it is noise."""
    codes, decisions = _select_candidates(
        [
            _candidate("J18.9", 0.99, "exact"),
            _candidate("J18", 0.80, "sparse"),
            _candidate("A15.0", 0.70, "ngram"),
        ],
        safety_bound=CANDIDATE_SAFETY_BOUND,
    )
    assert codes == ("J18.9",)
    assert any(d.reason == KEEP_EXACT for d in decisions if d.kept)
    assert all(not d.kept for d in decisions if d.code != "J18.9")


def test_selection_is_not_a_fixed_slice() -> None:
    """One strong candidate emits one; several comparable ones emit several."""
    one, _ = _select_candidates(
        [_candidate("A", 1.0, "sparse"), _candidate("B", 0.1, "sparse")],
        safety_bound=CANDIDATE_SAFETY_BOUND,
    )
    many, _ = _select_candidates(
        [_candidate(c, 0.9, "sparse") for c in "ABCDE"], safety_bound=CANDIDATE_SAFETY_BOUND
    )
    assert one == ("A",), "a weak candidate was admitted beside a strong one"
    assert len(many) == 5, "comparable candidates were truncated"


def test_the_maximum_is_a_safety_bound_not_the_rule() -> None:
    codes, decisions = _select_candidates(
        [_candidate(f"C{i:03d}", 0.9, "sparse") for i in range(40)], safety_bound=5
    )
    assert len(codes) == 5
    assert any(d.reason == DROP_SAFETY_BOUND for d in decisions)


def test_duplicate_codes_are_dropped_with_a_reason() -> None:
    codes, decisions = _select_candidates(
        [_candidate("X", 0.9, "sparse"), _candidate("X", 0.9, "sparse")],
        safety_bound=CANDIDATE_SAFETY_BOUND,
    )
    assert codes == ("X",)
    assert any(d.reason == DROP_DUPLICATE for d in decisions)


def test_no_candidates_yields_an_empty_list_not_an_invention() -> None:
    codes, decisions = _select_candidates([], safety_bound=CANDIDATE_SAFETY_BOUND)
    assert codes == ()
    assert decisions == ()


def test_every_candidate_carries_a_retain_or_remove_reason() -> None:
    _codes, decisions = _select_candidates(
        [
            _candidate("A", 0.9, "sparse"),
            _candidate("B", 0.2, "sparse"),
            _candidate("C", 0.5, "ngram"),
        ],
        safety_bound=CANDIDATE_SAFETY_BOUND,
    )
    assert decisions
    assert all(d.reason for d in decisions)


def test_real_pipeline_candidate_sizes_are_not_all_ten() -> None:
    result = run_document(FIX / "synthetic_medication_list.txt", _config(), mode="deterministic")
    payload = json.loads(to_submission_json(result.predictions))
    sizes = {len(e["candidates"]) for e in payload if "candidates" in e}
    assert sizes, "no candidate lists were produced"
    assert sizes != {10}, "candidate selection is still an unconditional top-10"


def test_decoding_is_deterministic_across_runs() -> None:
    first = run_document(FIX / "synthetic_medication_list.txt", _config(), mode="deterministic")
    second = run_document(FIX / "synthetic_medication_list.txt", _config(), mode="deterministic")
    assert to_submission_json(first.predictions) == to_submission_json(second.predictions)


def test_an_uncertain_assertion_is_not_forced_into_a_label() -> None:
    result = run_document(FIX / "synthetic_medical_document.txt", _config(), mode="deterministic")
    payload = json.loads(to_submission_json(result.predictions))
    for entity in payload:
        if "assertions" in entity:
            assert isinstance(entity["assertions"], list)


def test_decode_entities_emits_only_what_the_cascade_accepted() -> None:
    from mednorm_vi.confidence_cascade import CascadeDecision

    assert decode_entities((), (), (), ()) == ()
    # A rejecting cascade yields nothing even when a hypothesis exists.
    assert (
        decode_entities((), (CascadeDecision("h1", False, 0.0, ("below_threshold",)),), (), ())
        == ()
    )


# ---------------------------------------------------------------------------
# D. L9 KB membership
# ---------------------------------------------------------------------------


def _snapshots() -> LockedSnapshots:
    config = _config()
    return LockedSnapshots(
        icd10=load_index(config.icd_index), rxnorm=load_index(config.rxnorm_index)
    )


def _entity(organizer_type: str, candidates: list[str]) -> dict[str, object]:
    return {
        "text": "x",
        "type": organizer_type,
        "candidates": candidates,
        "assertions": [],
        "position": [0, 1],
    }


def test_real_pipeline_output_passes_kb_membership() -> None:
    result = run_document(FIX / "synthetic_medication_list.txt", _config(), mode="deterministic")
    payload = json.loads(to_submission_json(result.predictions))
    outcome = validate_document_candidates(payload, _snapshots(), document_id="d")
    assert outcome.ok, [i.message for i in outcome.errors]


def test_a_code_outside_the_snapshot_is_rejected() -> None:
    outcome = validate_entity_candidates(_entity("CHẨN_ĐOÁN", ["NOT_A_REAL_CODE"]), _snapshots())
    assert not outcome.ok
    assert any(i.code == "kb.candidate_not_in_snapshot" for i in outcome.errors)


def test_a_duplicate_candidate_is_rejected() -> None:
    outcome = validate_entity_candidates(_entity("THUỐC", ["198440", "198440"]), _snapshots())
    assert any(i.code == "kb.duplicate_candidate" for i in outcome.errors)


def test_candidates_on_a_type_without_an_ontology_are_rejected() -> None:
    """Spec §7.3: no ICD on MEDICATION, no RxNorm on DIAGNOSIS, none on SYMPTOM."""
    outcome = validate_entity_candidates(_entity("TRIỆU_CHỨNG", ["J18.9"]), _snapshots())
    assert any(i.code == "kb.candidates_on_type_without_ontology" for i in outcome.errors)


def test_the_wrong_snapshot_cannot_certify_membership() -> None:
    config = _config()
    swapped = LockedSnapshots(icd10=load_index(config.rxnorm_index))
    outcome = validate_entity_candidates(_entity("CHẨN_ĐOÁN", ["A000"]), swapped)
    assert any(i.code == "kb.wrong_snapshot" for i in outcome.errors)


def test_a_missing_snapshot_fails_rather_than_passing() -> None:
    outcome = validate_entity_candidates(_entity("CHẨN_ĐOÁN", ["A000"]), LockedSnapshots())
    assert any(i.code == "kb.snapshot_unavailable" for i in outcome.errors)


def test_a_code_not_offered_for_this_mention_is_rejected() -> None:
    """Spec P7: a model may SELECT from the offered set, never introduce a code."""
    outcome = validate_entity_candidates(
        _entity("CHẨN_ĐOÁN", ["A000"]), _snapshots(), offered_codes=["J18.9"]
    )
    assert any(i.code == "kb.candidate_not_offered" for i in outcome.errors)


def test_membership_validation_is_independent_of_the_retriever() -> None:
    """It takes only the snapshots and the output — no linker, no model."""
    import inspect

    import mednorm_vi.validator.kb_membership as module

    source = inspect.getsource(module)
    for forbidden in (
        "linking.icd10",
        "linking.rxnorm",
        "linking.snapshot",
        "llm",
        "transformers",
        "torch",
    ):
        assert forbidden not in source, (
            f"the final KB gate imports {forbidden}; it must not depend on the "
            "component whose bug it exists to catch"
        )


def test_l9_organizer_validation_still_passes_alongside_membership() -> None:
    result = run_document(FIX / "synthetic_laboratory.txt", _config(), mode="deterministic")
    payload = json.loads(to_submission_json(result.predictions))
    assert validate_organizer_document(payload, document_id="lab").ok
    assert validate_document_candidates(payload, _snapshots(), document_id="lab").ok


def test_an_entity_with_no_candidates_needs_no_snapshot() -> None:
    """An empty candidate list makes no membership claim, so it cannot fail one.

    Requiring a snapshot to certify `[]` would reject a legitimate `deterministic`
    run configured with no KB — a false positive. The codes-present rule is
    unchanged and `test_a_missing_snapshot_fails_rather_than_passing` holds it.
    """
    outcome = validate_entity_candidates(_entity("THUỐC", []), LockedSnapshots())
    assert outcome.ok, [i.message for i in outcome.errors]


def _packaging_config(tmp_path: Path, *, expected_documents: int) -> PipelineConfig:
    """The real profile with only `expected_documents` narrowed to the fixture count.

    Relative paths stay repo-relative, so the real locked indices are used — this
    test must not validate against a toy snapshot.
    """
    import yaml

    payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    payload["expected_documents"] = expected_documents
    target = tmp_path / "pipeline.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return PipelineConfig.load(target)


def test_packaging_stops_when_an_injected_code_is_outside_the_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canonical packaging path must refuse to write a violating submission.

    This is the integration counterpart to the unit checks above. It injects the
    bad code **upstream**, at the linker, which is exactly the failure mode the gate
    exists for: L9 must catch a linker/reranker/selector defect rather than trust the
    component that produced the code. It then drives the real
    `run_input_dir` -> serialize -> validate -> package path and asserts that
    packaging stops and nothing is written.
    """
    from mednorm_vi.inference import pipeline as pipeline_module
    from mednorm_vi.linking.models import LinkerResult
    from mednorm_vi.validator.final_gate import FinalValidationError

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # The medication fixture, because it actually yields linked THUỐC entities —
    # a document that produces no candidate could not exercise the gate at all.
    (input_dir / "1.txt").write_text(
        (FIX / "synthetic_medication_list.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )
    config = _packaging_config(tmp_path, expected_documents=1)
    output_zip = tmp_path / "output.zip"

    # Control: the same run packages successfully when nothing is injected.
    run_input_dir(input_dir, output_zip=output_zip, config=config, mode="deterministic")
    assert output_zip.is_file(), "the control run must produce a package"
    output_zip.unlink()
    shutil.rmtree(tmp_path / "output")

    forged = "999999999999"
    assert not load_index(config.rxnorm_index).exists(forged), (
        "the forged code must genuinely be absent from the locked snapshot"
    )

    def _forging_linker(hypothesis, index):  # type: ignore[no-untyped-def]
        return LinkerResult(
            mention_id=hypothesis.hypothesis_id,
            candidates=(LinkedCandidate(code=forged, score=0.99, channels=("injected",)),),
        )

    monkeypatch.setattr(pipeline_module, "link_rxnorm", _forging_linker)

    with pytest.raises(FinalValidationError) as raised:
        run_input_dir(input_dir, output_zip=output_zip, config=config, mode="deterministic")

    assert any("kb.candidate_not_in_snapshot" in v for v in raised.value.violations)
    assert not output_zip.exists(), "a violating run must not write output.zip"
    assert not (tmp_path / "output").exists(), (
        "a violating run must not leave an output directory behind either"
    )


def test_the_canonical_packaging_path_actually_invokes_the_membership_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards against the gate being silently unwired again.

    Before Audit 0053's correction pass, `validate_document_candidates` had **no
    caller under `inference/`**: the validator existed and nothing ran it. A test
    that only calls it directly would not have noticed. Audit 0056a moved the
    canonical final boundary to `validator.final_gate`, so this test patches that
    boundary rather than recreating the old `inference.pipeline` call graph.
    """
    from mednorm_vi.validator import final_gate as final_gate_module
    from mednorm_vi.validator.final_gate import FinalValidationError
    from mednorm_vi.validator.results import (
        Severity,
        ValidationIssue,
        ValidationResult,
    )

    calls: list[str] = []
    original = final_gate_module.validate_document_candidates

    def _recording(root, snapshots, **kwargs):  # type: ignore[no-untyped-def]
        document_id = str(kwargs.get("document_id"))
        calls.append(document_id)
        result = original(root, snapshots, **kwargs)
        if document_id == "2":
            return result.merged_with(
                ValidationResult.from_issues(
                    [
                        ValidationIssue(
                            code="kb.injected_membership_probe",
                            message="membership validator was reached by the final gate",
                            severity=Severity.ERROR,
                            document_id=document_id,
                        )
                    ]
                )
            )
        return result

    monkeypatch.setattr(final_gate_module, "validate_document_candidates", _recording)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text(
        (FIX / "synthetic_medication_list.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (input_dir / "2.txt").write_text("PLT: 250\n", encoding="utf-8")
    output_zip = tmp_path / "output.zip"
    with pytest.raises(FinalValidationError) as raised:
        run_input_dir(
            input_dir,
            output_zip=output_zip,
            config=_packaging_config(tmp_path, expected_documents=2),
            mode="deterministic",
        )
    assert calls == ["1", "2"], (
        "the packaging path must validate every document's candidates, in order"
    )
    assert any(
        violation == "2.json:kb.injected_membership_probe" for violation in raised.value.violations
    )
    assert not output_zip.exists()
    assert not (tmp_path / "output").exists()


# ---------------------------------------------------------------------------
# E. Reproducibility files: static contract only (never built)
# ---------------------------------------------------------------------------


def test_the_lock_file_pins_every_requirement_exactly() -> None:
    text = (REPO / "requirements.lock").read_text(encoding="utf-8")
    pins = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    assert pins, "the lock file declares nothing"
    for pin in pins:
        assert "==" in pin, f"unpinned requirement: {pin!r}"
    for required in ("PyYAML==", "torch==", "transformers=="):
        assert any(p.startswith(required) for p in pins), f"missing pin {required}"


def test_the_dockerfile_is_offline_and_single_entry_point() -> None:
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "requirements.lock" in text
    assert text.count("ENTRYPOINT") == 1, "exactly one entry point is allowed"
    assert "mednorm_vi.inference.cli" in text, "the entry point must be the canonical CLI"
    assert "USER mednorm" in text, "the runtime must not be root"


def test_the_dockerfile_copies_no_weights_or_data() -> None:
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    copies = [line for line in text.splitlines() if line.startswith("COPY ")]
    assert copies
    for line in copies:
        for forbidden in ("checkpoint", "models/", "data/", "indices/", "notebooks"):
            assert forbidden not in line, f"Dockerfile copies {forbidden}: {line!r}"


def test_the_dockerfile_documents_the_required_mounts() -> None:
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    for mount in (
        "/models/e3",
        "/models/hf",
        "/models/vncorenlp",
        "/kb/indices",
        "/input",
        "/output",
    ):
        assert mount in text, f"undocumented mount point {mount}"


def test_no_weights_are_tracked_in_git() -> None:
    import subprocess

    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True)
    for line in tracked.splitlines():
        assert not line.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".bin"))
