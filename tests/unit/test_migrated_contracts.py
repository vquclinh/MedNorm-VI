"""Contracts migrated out of the ZS0 and E4 packages (Audit 0051).

Audit 0051 deleted `mednorm_vi.zs0` and `mednorm_vi.training.phase2.e4`. Several
pieces inside them were never specific to a baseline arm or to one expert, so they
were moved to architecture-aligned homes rather than deleted with their containers:

    zs0/backends.py      -> llm/backends.py, llm/structured_output.py
    zs0/proposals.py     -> mention_factory/offsets.py (+ gliner label map)
    zs0/linking.py       -> linking/snapshot.py
    zs0/assertions.py    -> specialists/assertion/cues.py
    zs0/ledger.py        -> governance/parameter_budget.py (concurrency, methods)
    e4/contracts.py      -> training/governed_splits.py (split identity)
    e4/w2ner.py          -> mention_factory/spans.py (EntitySpan, type order)

These tests are the surviving behavioural coverage for that code. They were carried
across from `tests/unit/test_zs0_baseline.py` and re-pointed at the new modules; the
assertions about ZS0 *arms*, ZS0 *arm selection* and the ZS0 *package validator*
were not carried across, because those described an abandoned workflow rather than
production behaviour — and in the packager's case, a validator that contradicted the
organizer layout this repository has confirmed since Audit 0002.

No model is loaded anywhere in this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mednorm_vi.governance.parameter_budget import (
    MAX_DEPLOYMENT_PARAMETERS,
    METHOD_COUNTED_FROM_CONFIG,
    METHOD_COUNTED_FROM_SAFETENSORS_INDEX,
    METHOD_PUBLISHED_ESTIMATE,
    STATUS_IMPLEMENTED,
    STATUS_TRAINED,
    CandidateModel,
    CandidateRegistry,
    UnverifiedDeploymentComponent,
    compute_deployment_budget,
)
from mednorm_vi.linking.snapshot import (
    SOURCE_ALIAS_EXACT,
    Candidate,
    OntologySnapshot,
    SnapshotLinkingError,
    assert_codes_in_snapshot,
    constrain_selection,
    generate_candidates,
    link_against_snapshot,
    normalize_alias,
    ontology_for_type,
)
from mednorm_vi.llm.backends import (
    LOCAL_FILES_ONLY,
    BackendError,
    CausalLMBackend,
    LocalCheckpointMissing,
    OpenTypeSpanBackend,
    assert_no_external_api,
    require_local_directory,
)
from mednorm_vi.llm.structured_output import (
    StructuredOutputError,
    extract_json_object,
    normalize_json_completion,
    require_json_object,
)
from mednorm_vi.mention_factory.offsets import (
    REJECT_AMBIGUOUS_OCCURRENCE,
    REJECT_ANCHOR_NOT_FOUND,
    REJECT_EMPTY_SPAN,
    REJECT_NOT_A_SUBSTRING,
    REJECT_OFFSET_MISMATCH,
    REJECT_OUT_OF_RANGE,
    REJECTION_REASONS,
    ProposalRejected,
    RejectionLedger,
    resolve_in_document,
    resolve_occurrence,
    verify_span,
)
from mednorm_vi.mention_factory.spans import (
    DEFAULT_TYPE_ORDER,
    EntitySpan,
    SpanContractError,
    assert_type_order_complete,
)
from mednorm_vi.specialists.assertion.cues import (
    CUES_BY_LABEL,
    DEFAULT_SCOPE_CHARACTERS,
    IS_FAMILY,
    IS_HISTORICAL,
    IS_NEGATED,
    CueContractError,
    CueScopeDecision,
    build_adjudication_payload,
    constrain_adjudication,
    decide_from_cues,
    find_cues,
)
from mednorm_vi.training.governed_splits import (
    FORBIDDEN_SPLIT_NAMES,
    GOVERNED_TRAIN_SHA256,
    GOVERNED_VALIDATION_SHA256,
    GovernedSplitError,
    assert_split_allowed,
    resolve_governed_split_by_sha256,
)

# ---------------------------------------------------------------------------
# A. Span contract (from the deleted W2NER module)
# ---------------------------------------------------------------------------


def test_entity_span_enforces_the_spec_section_4_invariant() -> None:
    text = "Bệnh nhân sốt cao"
    start = text.index("sốt")
    EntitySpan(start, start + 3, "SYMPTOM", "sốt").validate_against(text)
    with pytest.raises(SpanContractError, match="exact original_text slice"):
        EntitySpan(0, 3, "SYMPTOM", "sốt").validate_against(text)
    with pytest.raises(SpanContractError, match="unsupported entity type"):
        EntitySpan(start, start + 3, "NOT_A_TYPE", "sốt").validate_against(text)
    with pytest.raises(SpanContractError, match="invalid entity offsets"):
        EntitySpan(5, 5, "SYMPTOM", "").validate_against(text)


def test_the_type_order_axis_covers_exactly_the_five_types() -> None:
    assert_type_order_complete(DEFAULT_TYPE_ORDER)
    assert len(DEFAULT_TYPE_ORDER) == 5
    with pytest.raises(SpanContractError):
        assert_type_order_complete(DEFAULT_TYPE_ORDER[:4])
    with pytest.raises(SpanContractError):
        assert_type_order_complete((*DEFAULT_TYPE_ORDER, "DIAGNOSIS"))


# ---------------------------------------------------------------------------
# B. Offset verification and substring resolution
# ---------------------------------------------------------------------------


def test_verify_span_names_each_way_the_invariant_can_fail() -> None:
    text = "Paracetamol 500mg"
    verify_span(text, 0, 11, "Paracetamol")
    for start, end, claimed, reason in (
        (5, 5, "", REJECT_EMPTY_SPAN),
        (8, 3, "x", REJECT_EMPTY_SPAN),
        (0, 999, text, REJECT_OUT_OF_RANGE),
        (-1, 4, "Para", REJECT_OUT_OF_RANGE),
        (0, 11, "Ibuprofen__", REJECT_OFFSET_MISMATCH),
    ):
        with pytest.raises(ProposalRejected) as excinfo:
            verify_span(text, start, end, claimed)
        assert excinfo.value.reason == reason


def test_a_single_occurrence_resolves() -> None:
    segment = "Bệnh nhân ho khan nhiều"
    start, end = resolve_occurrence(segment, "ho khan")
    assert segment[start:end] == "ho khan"


def test_a_repeated_surface_form_fails_closed_without_an_anchor() -> None:
    # spec section 5 case C7: the same text at two offsets is two entities, so
    # guessing which one a text-only source meant would fabricate evidence.
    segment = "sốt cao rồi sốt nhẹ"
    with pytest.raises(ProposalRejected) as excinfo:
        resolve_occurrence(segment, "sốt")
    assert excinfo.value.reason == REJECT_AMBIGUOUS_OCCURRENCE


def test_an_anchor_disambiguates_exactly_one_occurrence() -> None:
    segment = "sốt cao rồi sốt nhẹ"
    start, end = resolve_occurrence(segment, "sốt", anchor="sốt nhẹ")
    assert segment[start:end] == "sốt"
    assert start == segment.index("sốt nhẹ")


def test_an_anchor_that_is_absent_or_repeated_is_refused() -> None:
    segment = "sốt cao rồi sốt nhẹ"
    with pytest.raises(ProposalRejected) as excinfo:
        resolve_occurrence(segment, "sốt", anchor="không có ở đây")
    assert excinfo.value.reason == REJECT_ANCHOR_NOT_FOUND
    # An anchor containing the mention twice resolves nothing.
    with pytest.raises(ProposalRejected) as excinfo:
        resolve_occurrence("a b a b", "a", anchor="a b a")
    assert excinfo.value.reason == REJECT_AMBIGUOUS_OCCURRENCE


def test_a_mention_absent_from_the_segment_is_refused() -> None:
    with pytest.raises(ProposalRejected) as excinfo:
        resolve_occurrence("ho khan", "viêm phổi")
    assert excinfo.value.reason == REJECT_NOT_A_SUBSTRING
    with pytest.raises(ProposalRejected):
        resolve_occurrence("ho khan", "")


def test_segment_offsets_are_reverified_in_document_coordinates() -> None:
    document = "Tiền sử: sốt. Hiện tại: ho khan."
    segment = "Hiện tại: ho khan."
    segment_start = document.index(segment)
    start, end = resolve_in_document(
        original_text=document, segment_start=segment_start,
        segment_text=segment, mention="ho khan")
    assert document[start:end] == "ho khan"
    # A wrong segment offset lands wrong in the document and must be caught there.
    with pytest.raises(ProposalRejected) as excinfo:
        resolve_in_document(
            original_text=document, segment_start=0,
            segment_text=segment, mention="ho khan")
    assert excinfo.value.reason == REJECT_OFFSET_MISMATCH


def test_the_rejection_ledger_counts_reasons_and_holds_no_clinical_text() -> None:
    ledger = RejectionLedger()
    ledger.record(REJECT_AMBIGUOUS_OCCURRENCE)
    ledger.record(REJECT_AMBIGUOUS_OCCURRENCE)
    ledger.record(REJECT_OFFSET_MISMATCH)
    assert ledger.total == 3
    payload = ledger.as_dict()
    assert payload["by_reason"][REJECT_AMBIGUOUS_OCCURRENCE] == 2
    assert payload["contains_clinical_text"] is False
    assert set(payload["by_reason"]) == set(REJECTION_REASONS)
    with pytest.raises(ValueError, match="unknown rejection reason"):
        ledger.record("invented_reason")


# ---------------------------------------------------------------------------
# C. Snapshot-grounded linking (spec P7)
# ---------------------------------------------------------------------------


def _icd_snapshot() -> OntologySnapshot:
    return OntologySnapshot(
        ontology="ICD10", snapshot_id="tt06-2026",
        codes=frozenset({"J18.9", "J18", "A15.0"}),
        aliases={normalize_alias("Viêm phổi"): ("J18.9",)})


def test_a_type_only_takes_its_own_ontology() -> None:
    assert ontology_for_type("DIAGNOSIS") == "ICD10"
    assert ontology_for_type("MEDICATION") == "RXNORM"
    assert ontology_for_type("SYMPTOM") is None
    with pytest.raises(SnapshotLinkingError, match="cross-ontology"):
        generate_candidates(
            mention_text="Paracetamol", entity_type="MEDICATION",
            snapshot=_icd_snapshot())


def test_a_type_carrying_no_candidates_returns_none() -> None:
    result = link_against_snapshot(
        mention_text="ho khan", entity_type="SYMPTOM", snapshot=_icd_snapshot())
    assert result.candidates == ()
    assert result.fallback == "type_carries_no_candidates"


def test_an_exact_alias_hit_wins_outright() -> None:
    result = link_against_snapshot(
        mention_text="viêm  phổi", entity_type="DIAGNOSIS",
        snapshot=_icd_snapshot(),
        dense=[("A15.0", 0.99)])
    assert result.fallback == "exact_alias"
    assert result.codes() == ("J18.9",)
    assert result.candidates[0].source == SOURCE_ALIAS_EXACT


def test_retrieval_is_filtered_by_snapshot_membership_as_it_is_read() -> None:
    # A drifted index offering a code the snapshot does not contain contributes
    # nothing, rather than being caught only at the final check.
    result = link_against_snapshot(
        mention_text="phổi", entity_type="DIAGNOSIS", snapshot=_icd_snapshot(),
        lexical=[("NOT_IN_SNAPSHOT", 0.99)], dense=[("J18", 0.4)])
    assert result.codes() == ("J18",)


def test_no_candidate_retrieved_means_empty_not_invented() -> None:
    result = link_against_snapshot(
        mention_text="không rõ", entity_type="DIAGNOSIS", snapshot=_icd_snapshot())
    assert result.candidates == ()
    assert result.fallback == "no_candidate_retrieved"
    assert result.as_dict()["codes_were_generated_by_a_model"] is False


def test_a_model_may_only_select_from_the_offered_set() -> None:
    snapshot = _icd_snapshot()
    offered = (
        Candidate(code="J18", ontology="ICD10", score=0.6, source="dense", rank=1),
        Candidate(code="A15.0", ontology="ICD10", score=0.4, source="dense", rank=2),
    )
    selected, rejected = constrain_selection(
        '{"selected": ["A15.0"]}', offered, snapshot=snapshot)
    assert [c.code for c in selected] == ["A15.0"]
    assert rejected == ()

    # "J18.9" EXISTS in the snapshot but was not offered for this mention: it is
    # dropped anyway, because it was not retrieved as evidence for this mention.
    selected, rejected = constrain_selection(
        '{"selected": ["J18.9"]}', offered, snapshot=snapshot)
    assert selected == ()
    assert rejected == ("J18.9",)


def test_a_malformed_selection_falls_back_to_the_retrieved_order() -> None:
    snapshot = _icd_snapshot()
    offered = (
        Candidate(code="J18", ontology="ICD10", score=0.6, source="dense", rank=1),)
    for payload in ("not json at all", '{"selected": "J18"}', "{}"):
        selected, rejected = constrain_selection(payload, offered, snapshot=snapshot)
        assert [c.code for c in selected] == ["J18"]
        assert rejected == ()


def test_every_emitted_code_is_rechecked_against_the_snapshot() -> None:
    snapshot = _icd_snapshot()
    assert_codes_in_snapshot(["J18", "A15.0"], snapshot=snapshot)
    with pytest.raises(SnapshotLinkingError, match="retrieved, never generated"):
        assert_codes_in_snapshot(["J99.9"], snapshot=snapshot)


def test_an_unknown_retrieval_source_is_refused() -> None:
    with pytest.raises(SnapshotLinkingError, match="unknown retrieval source"):
        Candidate(code="J18", ontology="ICD10", score=0.1, source="vibes", rank=1)


# ---------------------------------------------------------------------------
# D. Assertion cue and scope evidence (spec section 8)
# ---------------------------------------------------------------------------


def test_a_cue_in_scope_produces_its_label() -> None:
    text = "Bệnh nhân không có tiểu đường"
    decision = decide_from_cues(text, mention_start=text.index("tiểu đường"))
    assert IS_NEGATED in decision.labels
    assert decision.uncertain is False
    assert decision.source == "deterministic_cue_in_scope"


def test_no_cue_anywhere_is_a_confident_empty_set() -> None:
    text = "Chẩn đoán: viêm phổi"
    decision = decide_from_cues(text, mention_start=text.index("viêm phổi"))
    assert decision.labels == ()
    assert decision.uncertain is False
    assert decision.source == "deterministic_no_cue"


def test_a_cue_out_of_scope_decides_nothing_and_reports_uncertainty() -> None:
    # spec section 4.3: "keyword presence must never label every entity
    # indiscriminately". A distant cue routes to adjudication; it does not label.
    text = "không " + ("x" * (DEFAULT_SCOPE_CHARACTERS + 20)) + " viêm phổi"
    decision = decide_from_cues(text, mention_start=text.index("viêm phổi"))
    assert decision.labels == ()
    assert decision.uncertain is True
    assert decision.source == "deterministic_out_of_scope"
    assert decision.evidence and all(not e.within_scope for e in decision.evidence)


def test_an_entity_may_carry_several_assertions() -> None:
    text = "Tiền sử gia đình không có ung thư"
    decision = decide_from_cues(text, mention_start=text.index("ung thư"))
    assert len(decision.labels) >= 2
    assert set(decision.labels) <= {IS_NEGATED, IS_FAMILY, IS_HISTORICAL}


def test_cues_are_found_before_the_mention_and_sorted_deterministically() -> None:
    text = "không sốt"
    first = find_cues(text, mention_start=text.index("sốt"))
    assert first == find_cues(text, mention_start=text.index("sốt"))
    # A cue AFTER the mention is not evidence for it.
    assert find_cues("sốt không", mention_start=0) == ()


def test_an_adjudicator_cannot_invent_a_label() -> None:
    baseline = CueScopeDecision(labels=(IS_NEGATED,), source="deterministic_cue_in_scope")
    ok = constrain_adjudication('{"assertions": ["isHistorical"]}', deterministic=baseline)
    assert ok.labels == (IS_HISTORICAL,)
    assert ok.source == "adjudicated"
    for payload, expected_source in (
        ("not json", "adjudicator_rejected_malformed_json"),
        ('{"assertions": "isNegated"}', "adjudicator_rejected_malformed_record"),
        ('{"assertions": [7]}', "adjudicator_rejected_non_string_label"),
        ('{"assertions": ["isProbablyFine"]}', "adjudicator_rejected_unsupported_label"),
    ):
        fallback = constrain_adjudication(payload, deterministic=baseline)
        assert fallback.source == expected_source
        assert fallback.labels == baseline.labels
        assert fallback.uncertain is True


def test_an_unsupported_label_cannot_be_constructed() -> None:
    with pytest.raises(CueContractError, match="unsupported assertion labels"):
        CueScopeDecision(labels=("isDefinitelyTrue",), source="x")


def test_the_adjudication_payload_forbids_touching_the_span_or_type() -> None:
    payload = build_adjudication_payload(
        mention_text="ung thư", entity_type="DIAGNOSIS", context="…",
        route="narrative",
        deterministic=CueScopeDecision(labels=(), source="deterministic_no_cue"))
    assert payload["may_modify_span"] is False
    assert payload["may_modify_type"] is False
    assert payload["may_produce_free_text"] is False
    assert sorted(payload["allowed_labels"]) == [IS_FAMILY, IS_HISTORICAL, IS_NEGATED]


def test_the_hydra_uses_the_shared_cue_lexicons() -> None:
    """One source of truth for the cue families across L5."""
    from mednorm_vi.specialists.assertion import hydra

    assert hydra.NEGATION_CUES is CUES_BY_LABEL[IS_NEGATED]
    assert hydra.FAMILY_CUES is CUES_BY_LABEL[IS_FAMILY]
    assert hydra.HISTORICAL_CUES is CUES_BY_LABEL[IS_HISTORICAL]


# ---------------------------------------------------------------------------
# E. Local-only backends: lazy, fail-closed, never downloading
# ---------------------------------------------------------------------------


def test_the_local_only_flag_is_on_and_is_a_constant() -> None:
    assert LOCAL_FILES_ONLY is True


def test_a_backend_with_no_local_path_fails_closed() -> None:
    for backend in (OpenTypeSpanBackend(local_path=""), CausalLMBackend(local_path="")):
        with pytest.raises(LocalCheckpointMissing, match="no configured local path"):
            backend.load()


def test_a_backend_with_a_missing_local_path_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "not-materialized"
    for backend in (OpenTypeSpanBackend(local_path=str(missing)),
                    CausalLMBackend(local_path=str(missing))):
        with pytest.raises(LocalCheckpointMissing) as excinfo:
            backend.load()
        # The error must name the exact path; "model not found" is unactionable.
        assert str(missing) in str(excinfo.value)
        assert backend.loaded is False


def test_require_local_directory_refuses_a_file_and_accepts_a_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "weights.bin").write_bytes(b"x")
    with pytest.raises(LocalCheckpointMissing):
        require_local_directory(tmp_path / "weights.bin", role="test")
    assert require_local_directory(tmp_path, role="test") == tmp_path


def test_an_unloaded_backend_refuses_to_predict_or_count() -> None:
    span_backend = OpenTypeSpanBackend(local_path="anywhere")
    text_backend = CausalLMBackend(local_path="anywhere")
    with pytest.raises(BackendError, match="not loaded"):
        span_backend.predict("sốt")
    with pytest.raises(BackendError, match="cannot count parameters"):
        span_backend.parameter_count()
    with pytest.raises(BackendError, match="not loaded"):
        text_backend.generate("prompt")
    with pytest.raises(BackendError, match="cannot count parameters"):
        text_backend.parameter_count()


def test_an_external_api_is_refused_by_name() -> None:
    assert_no_external_api({"local_model_root": "/models"})
    for key in ("openai_api_key", "anthropic_api_key", "external_api_base",
                "remote_inference_endpoint"):
        with pytest.raises(BackendError, match="self-hosted"):
            assert_no_external_api({key: "set"})


# ---------------------------------------------------------------------------
# F. Constrained structured output
# ---------------------------------------------------------------------------


def test_a_fenced_or_chatty_completion_still_yields_its_object() -> None:
    assert extract_json_object('Sure! ```json\n{"a": 1}\n``` done') == '{"a": 1}'
    assert normalize_json_completion('  {"a": 1}  ') == '{"a": 1}'
    assert require_json_object('here you go: {"selected": []}') == {"selected": []}


def test_nesting_and_braces_inside_strings_do_not_confuse_extraction() -> None:
    payload = '{"a": {"b": "}"}, "c": "\\""}'
    assert extract_json_object(f"noise {payload} more") == payload


def test_a_completion_with_no_object_is_returned_unchanged_for_the_caller_to_reject() -> None:
    assert extract_json_object("no braces here") == "no braces here"
    assert normalize_json_completion("no braces here") == "no braces here"


def test_extraction_is_not_repair() -> None:
    with pytest.raises(StructuredOutputError, match="not well-formed JSON"):
        require_json_object('{"unclosed": ')
    with pytest.raises(StructuredOutputError, match="not an object"):
        require_json_object("[1, 2, 3]")


# ---------------------------------------------------------------------------
# G. Governed split identity (from the deleted E4 contracts)
# ---------------------------------------------------------------------------


def test_internal_test_can_never_be_resolved() -> None:
    assert "internal_test" in FORBIDDEN_SPLIT_NAMES
    assert assert_split_allowed("validation") == "validation"
    with pytest.raises(GovernedSplitError, match="held out"):
        assert_split_allowed("internal_test")
    with pytest.raises(GovernedSplitError):
        resolve_governed_split_by_sha256(
            split="internal_test", expected_sha256="0" * 64, search_roots=[])


def test_a_split_is_identified_by_digest_not_by_name(tmp_path: Path) -> None:
    import hashlib

    payload = '{"text": "sốt"}\n'.encode()
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "misleadingly_named.jsonl").write_bytes(payload)
    resolved = resolve_governed_split_by_sha256(
        split="validation", expected_sha256=digest, search_roots=[tmp_path])
    assert resolved.sha256 == digest
    assert resolved.path.name == "misleadingly_named.jsonl"


def test_a_digest_that_matches_nothing_raises_and_names_the_search(tmp_path: Path) -> None:
    (tmp_path / "other.jsonl").write_bytes(b"{}\n")
    with pytest.raises(FileNotFoundError, match="could not locate governed"):
        resolve_governed_split_by_sha256(
            split="train", expected_sha256="f" * 64, search_roots=[tmp_path])


def test_the_governed_digests_are_the_recorded_ones() -> None:
    # Carried across verbatim from the deleted E4 contract module; every S1/E5
    # measurement in the audits was computed against these exact bytes.
    assert GOVERNED_TRAIN_SHA256 == (
        "892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a")
    assert GOVERNED_VALIDATION_SHA256 == (
        "ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103")


# ---------------------------------------------------------------------------
# H. Deployment ledger behaviour migrated from the ZS0 ledger
# ---------------------------------------------------------------------------


def _component(component_id: str, **kwargs: object) -> CandidateModel:
    defaults: dict[str, object] = {
        "architecture_layer": "L7", "training_stage": "-",
        "status": STATUS_IMPLEMENTED, "total_parameters": 1_000_000_000,
        "parameter_count_method": METHOD_COUNTED_FROM_CONFIG,
        "parameter_count_verified": True, "loaded_at_inference": True,
    }
    defaults.update(kwargs)
    return CandidateModel(component_id=component_id, **defaults)  # type: ignore[arg-type]


def test_a_safetensors_index_count_is_a_verified_method() -> None:
    component = _component(
        "dense", parameter_count_method=METHOD_COUNTED_FROM_SAFETENSORS_INDEX)
    assert component.has_verified_count


def test_a_published_estimate_still_fails_closed() -> None:
    registry = CandidateRegistry(components=(
        _component("guessed", parameter_count_method=METHOD_PUBLISHED_ESTIMATE,
                   parameter_count_verified=False),))
    with pytest.raises(UnverifiedDeploymentComponent, match="fails closed"):
        compute_deployment_budget(registry, ["guessed"])


def test_peak_concurrency_is_reported_apart_from_the_9b_total() -> None:
    """spec section 21: the full stack need not be resident simultaneously.

    Sequential load/unload does not reduce what counts against the 9B cap, but it
    does reduce peak residency — so conflating the two would either waste budget or
    misreport VRAM.
    """
    registry = CandidateRegistry(components=(
        _component("resident_a", concurrent_with_other_models=True),
        _component("resident_b", concurrent_with_other_models=True),
        _component("sequential", concurrent_with_other_models=False),
    ))
    report = compute_deployment_budget(
        registry, ["resident_a", "resident_b", "sequential"])
    assert report.total_loaded_parameters == 3_000_000_000
    assert report.concurrently_resident_parameters == 2_000_000_000
    assert report.concurrently_resident_parameters < report.total_loaded_parameters
    assert report.within_budget
    payload = report.as_dict()
    assert payload["concurrently_resident_parameters"] == 2_000_000_000
    assert payload["max_deployment_parameters"] == MAX_DEPLOYMENT_PARAMETERS


def test_a_shared_backbone_is_counted_once_and_a_lora_counts_base_plus_adapter() -> None:
    registry = CandidateRegistry(components=(
        _component("backbone_head_a", total_parameters=1_700_000_000,
                   shares_weights_with="qwen_base", adapter_parameters=5_000_000,
                   status=STATUS_TRAINED),
        _component("backbone_head_b", total_parameters=1_700_000_000,
                   shares_weights_with="qwen_base", adapter_parameters=7_000_000,
                   status=STATUS_TRAINED),
    ))
    report = compute_deployment_budget(
        registry, ["backbone_head_a", "backbone_head_b"])
    # Base once (1.7B) + its adapter (5M) + the second head's adapter only (7M).
    assert report.total_loaded_parameters == 1_700_000_000 + 5_000_000 + 7_000_000
    assert report.components[1].counted_once is False
    assert report.as_dict()["adapter_parameters"] == 12_000_000
