"""Milestone 0081 guards: contextual proposal, finite verification, assertions.

No model weights, no network. The Qwen passes are injected fakes so the same production
orchestration path runs end to end on a laptop.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]

from mednorm_vi.contextual import aliases as ca  # noqa: E402
from mednorm_vi.contextual import assertions as cs  # noqa: E402
from mednorm_vi.contextual import document as cd  # noqa: E402
from mednorm_vi.contextual import lattice as cl  # noqa: E402
from mednorm_vi.contextual import proposer as cp  # noqa: E402
from mednorm_vi.contextual import resolver as cr  # noqa: E402
from mednorm_vi.contextual import runtime as crt  # noqa: E402
from mednorm_vi.contextual import verifier as cv  # noqa: E402
from mednorm_vi.contextual.proposals import (  # noqa: E402
    SOURCE_ALIAS,
    SOURCE_E3,
    SOURCE_QWEN,
    Proposal,
    ProposalPool,
    propose,
    propose_from_offsets,
)

CLI_PATH = ROOT / "scripts/contextual_0081.py"
_spec = importlib.util.spec_from_file_location("contextual_cli", CLI_PATH)
assert _spec and _spec.loader
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

TYPE_SYMPTOM = "TRIỆU_CHỨNG"
TYPE_DIAGNOSIS = "CHẨN_ĐOÁN"
TYPE_TEST = "TÊN_XÉT_NGHIỆM"
TYPE_RESULT = "KẾT_QUẢ_XÉT_NGHIỆM"
TYPE_DRUG = "THUỐC"


def view(source: str) -> cd.DocumentView:
    return cd.DocumentView.build(source)


def span(document: cd.DocumentView, text: str, entity_type: str, *,
         sources: set[str] | None = None, occurrence: int = 0) -> Proposal:
    start = -1
    for _ in range(occurrence + 1):
        start = document.source.index(text, start + 1)
    return Proposal(
        start=start, end=start + len(text), text=text, type=entity_type,
        sources=frozenset(sources or {SOURCE_E3}),
    )


# ------------------------------------------------------------------ line-addressed document


def test_line_ids_are_stable_and_map_to_real_offsets() -> None:
    document = view("dòng một\ndòng hai\ndòng ba")
    assert [line.line_id for line in document.lines] == ["L001", "L002", "L003"]
    for line in document.lines:
        assert document.source[line.start : line.end] == line.text
    assert "L002| dòng hai" in document.render()


def test_alignment_resolves_offsets_from_the_source_not_from_the_model() -> None:
    document = view("Khám: bệnh nhân sốt cao\nXử trí: hạ sốt")
    alignment = cd.align(document, "L001", "sốt cao")
    assert alignment.ok
    assert document.source[alignment.start : alignment.end] == "sốt cao"
    assert alignment.how == cd.ALIGN_LINE


def test_text_absent_from_the_source_is_rejected() -> None:
    document = view("bệnh nhân sốt")
    assert not cd.align(document, "L001", "viêm phổi thùy dưới").ok
    assert cd.align(document, "L001", "viêm phổi thùy dưới").reason in {
        cd.REJECT_NOT_IN_SOURCE, cd.REJECT_UNKNOWN_LINE
    }
    assert not cd.align(document, "L001", "   ").ok


def test_repeated_mentions_at_distinct_locations_are_preserved() -> None:
    document = view("ho khan\nsau đó ho khan lại xuất hiện")
    first = cd.align(document, "L001", "ho khan")
    second = cd.align(document, "L002", "ho khan")
    assert first.ok and second.ok
    assert first.start != second.start
    assert document.source[second.start : second.end] == "ho khan"


def test_occurrence_selects_within_a_line_and_out_of_range_is_refused() -> None:
    document = view("ho khan, ho khan nhiều")
    assert cd.align(document, "L001", "ho khan", 1).start == document.source.index(
        "ho khan", 1
    )
    assert cd.align(document, "L001", "ho khan", 5).reason == cd.REJECT_OCCURRENCE


def test_a_wrong_line_id_falls_back_only_when_the_text_is_unique() -> None:
    document = view("dòng một\nsốt cao\ndòng ba")
    unique = cd.align(document, "L999", "sốt cao")
    assert unique.ok and unique.how == cd.ALIGN_UNIQUE and unique.line_id == "L002"

    ambiguous = view("sốt cao\nsốt cao")
    assert cd.align(ambiguous, "L999", "sốt cao").reason == cd.REJECT_AMBIGUOUS


# ------------------------------------------------------------------ governed alias proposals


def test_alias_matching_requires_whole_word_boundaries() -> None:
    lexicon = ca.AliasLexicon()
    assert lexicon.add("paracetamol", TYPE_DRUG)
    source = "dùng paracetamol 500mg"
    hits = ca.find_alias_hits(source, lexicon)
    assert [h.text for h in hits] == ["paracetamol"]
    assert source[hits[0].start : hits[0].end] == "paracetamol"


def test_mid_word_medication_match_is_rejected() -> None:
    lexicon = ca.AliasLexicon()
    lexicon.add("ceta", TYPE_DRUG)
    # "ceta" sits inside "paracetamol" - a substring, not a token
    assert ca.find_alias_hits("bệnh nhân dùng paracetamol", lexicon) == []


def test_short_lab_alias_inside_an_ordinary_word_is_rejected() -> None:
    lexicon = ca.AliasLexicon()
    lexicon.add("AST", TYPE_TEST)
    assert ca.find_alias_hits("bệnh nhân bị hen suyễn ASTma nặng", lexicon) == []
    assert ca.find_alias_hits("xét nghiệm AST tăng", lexicon)[0].text == "AST"
    # a short alias is case-disciplined: lowercase prose must not fire it
    assert ca.find_alias_hits("ast của bệnh nhân", lexicon) == []


def test_aliases_shorter_than_the_floor_are_never_registered() -> None:
    lexicon = ca.AliasLexicon()
    assert not lexicon.add("ho", TYPE_DIAGNOSIS)
    assert lexicon.skipped["single_token_too_short"] == 1
    assert lexicon.size == 0


def test_longest_governed_phrase_wins_over_its_own_fragment() -> None:
    lexicon = ca.AliasLexicon()
    lexicon.add("viêm phổi", TYPE_DIAGNOSIS)
    lexicon.add("viêm phổi thùy dưới", TYPE_DIAGNOSIS)
    hits = ca.find_alias_hits("chẩn đoán viêm phổi thùy dưới phải", lexicon)
    assert [h.text for h in hits] == ["viêm phổi thùy dưới"]


def test_lexicon_is_built_from_whatever_governed_index_is_supplied() -> None:
    class FakeIndex:
        records = {
            "A1": {"canonical_name": "viêm phổi", "aliases": ["viêm phổi cấp"]},
            "A2": {"canonical_name": "x" * 2},          # too short, skipped
        }

    lexicon = ca.build_lexicon([(FakeIndex(), TYPE_DIAGNOSIS)])
    assert lexicon.size == 2
    hits = ca.find_alias_hits("bệnh nhân viêm phổi cấp", lexicon)
    assert [h.text for h in hits] == ["viêm phổi cấp"]
    assert hits[0].entity_type == TYPE_DIAGNOSIS


# ------------------------------------------------------------------ proposer contract


def test_each_specialized_pass_teaches_its_own_distinctions() -> None:
    rendered = view("L1 text").render()
    diagnosis = cp.build_pass_prompt(cp.PASS_DIAGNOSIS, rendered)
    for taught in (
        "COMPLETE clinical concept", "A symptom is what the patient feels",
        "Do NOT split one concept", "Repeated mentions are not duplicates",
        "Never return a character position",
    ):
        assert taught in diagnosis, taught
    assert "THUỐC" not in diagnosis      # a pass asks for its own types only

    medication = cp.build_pass_prompt(cp.PASS_MEDICATION, rendered)
    for taught in (
        "NEVER return a substring of a longer medication name",
        "brand name, a generic name", "Stop at the end of that item",
    ):
        assert taught in medication, taught

    lab = cp.build_pass_prompt(cp.PASS_LAB, rendered)
    for taught in (
        "A RESULT BELONGS TO A TEST", "with no named measurement is NOT a result",
        "two separate entities",
    ):
        assert taught in lab, taught


def test_no_proposer_prompt_names_a_disease_drug_or_test() -> None:
    """The rubrics teach shape, never content. Nothing test-set specific may leak in."""
    for pass_name in cp.PASSES:
        prompt = cp.build_pass_prompt(pass_name, "L001| x").casefold()
        for leaked in ("paracetamol", "amlodipine", "viêm phổi", "g6pd", "metformin"):
            assert leaked not in prompt, f"{pass_name}: {leaked}"


def test_proposer_parse_is_strict_and_counts_bad_rows() -> None:
    result = cp.parse_pass(
        cp.PASS_DIAGNOSIS,
        '{"entities": [{"line_id": "L001", "text": "sốt", "type": "TRIỆU_CHỨNG", '
        '"occurrence": 0}, {"line_id": "L001", "text": "", "type": "TRIỆU_CHỨNG"}, '
        '{"line_id": "L001", "text": "x", "type": "NOT_A_TYPE"}]}'
    )
    assert [p.text for p in result.proposals] == ["sốt"]
    assert result.counters[cp.PARSE_BAD_ROW] == 2
    assert result.report.json_extracted

    failed = cp.parse_pass(cp.PASS_DIAGNOSIS, "I think there is a fever.")
    assert failed.proposals == []
    assert failed.report.category == cp.PARSE_NO_JSON


def test_hallucinated_proposal_text_never_becomes_a_proposal() -> None:
    document = view("bệnh nhân ho khan")
    pool = ProposalPool()
    propose(pool, document, source=SOURCE_QWEN, line_id="L001",
            text="viêm phổi nặng", entity_type=TYPE_DIAGNOSIS)
    assert pool.proposals == []
    assert sum(pool.rejected.values()) == 1


def test_every_stored_proposal_is_a_literal_source_slice() -> None:
    document = view("bệnh nhân ho khan nhiều")
    pool = ProposalPool()
    propose(pool, document, source=SOURCE_QWEN, line_id="L001",
            text="ho khan", entity_type=TYPE_SYMPTOM)
    propose_from_offsets(pool, document, source=SOURCE_E3, start=0, end=9,
                         entity_type=TYPE_SYMPTOM)
    for proposal in pool.proposals:
        assert document.source[proposal.start : proposal.end] == proposal.text


def test_identical_spans_from_two_sources_merge_into_one_agreed_proposal() -> None:
    document = view("bệnh nhân ho khan")
    pool = ProposalPool()
    propose(pool, document, source=SOURCE_QWEN, line_id="L001", text="ho khan",
            entity_type=TYPE_SYMPTOM)
    propose(pool, document, source=SOURCE_ALIAS, line_id="L001", text="ho khan",
            entity_type=TYPE_SYMPTOM)
    assert len(pool.proposals) == 1
    assert pool.proposals[0].sources == {SOURCE_QWEN, SOURCE_ALIAS}
    assert pool.proposals[0].agreed


# ------------------------------------------------------------------ verifier contract


def test_verifier_prompt_offers_only_indices_and_forbids_new_text() -> None:
    document = view("bệnh nhân viêm phổi nặng")
    group = cl.LatticeGroup(
        start=10, end=24,
        options=(span(document, "viêm phổi", TYPE_DIAGNOSIS),
                 span(document, "viêm phổi nặng", TYPE_DIAGNOSIS)),
    )
    prompt = cv.build_verification_prompt(document, group)
    assert "[0]" in prompt and "[1]" in prompt
    assert "You CANNOT write new text" in prompt
    assert "COMPLETE minimal clinical concept" in prompt


def test_verifier_cannot_invent_a_span() -> None:
    document = view("bệnh nhân viêm phổi")
    group = cl.LatticeGroup(
        start=10, end=19, options=(span(document, "viêm phổi", TYPE_DIAGNOSIS),)
    )
    verdicts, _ = cv.parse_verdicts(
        '{"verdicts": [{"index": 0, "action": "accept", "text": "viêm phổi thùy dưới"}]}', 1
    )
    result = cv.apply_verdicts(group, verdicts, parse_failed=False)
    assert [p.text for p in result.accepted] == ["viêm phổi"]   # the extra text is ignored


def test_an_invalid_index_fails_closed() -> None:
    verdicts, counters = cv.parse_verdicts(
        '{"verdicts": [{"index": 7, "action": "accept"}, {"index": -1, "action": "accept"}]}',
        2,
    )
    assert verdicts == []
    assert counters[cv.INVALID_INDEX] == 2


def test_an_unmentioned_candidate_is_not_accepted_by_silence() -> None:
    document = view("bệnh nhân ho khan nhiều")
    group = cl.LatticeGroup(
        start=10, end=23,
        options=(span(document, "ho khan", TYPE_SYMPTOM, sources={SOURCE_QWEN}),
                 span(document, "ho khan nhiều", TYPE_SYMPTOM, sources={SOURCE_QWEN})),
    )
    verdicts, _ = cv.parse_verdicts('{"verdicts": [{"index": 1, "action": "accept"}]}', 2)
    result = cv.apply_verdicts(group, verdicts, parse_failed=False)
    assert [p.text for p in result.accepted] == ["ho khan nhiều"]
    assert [p.text for p in result.rejected] == ["ho khan"]


def test_a_parse_failure_falls_back_to_the_trusted_seed_only() -> None:
    document = view("bệnh nhân ho khan nhiều")
    group = cl.LatticeGroup(
        start=10, end=23,
        options=(span(document, "ho khan", TYPE_SYMPTOM, sources={SOURCE_E3}),
                 span(document, "ho khan nhiều", TYPE_SYMPTOM, sources={SOURCE_QWEN})),
    )
    result = cv.apply_verdicts(group, [], parse_failed=True)
    assert [p.text for p in result.accepted] == ["ho khan"]
    assert [p.text for p in result.rejected] == ["ho khan nhiều"]


def test_retype_is_restricted_to_the_five_organizer_types() -> None:
    verdicts, counters = cv.parse_verdicts(
        '{"verdicts": [{"index": 0, "action": "retype", "type": "DISEASE"}]}', 1
    )
    assert verdicts == [] and counters[cv.INVALID_TYPE] == 1

    good, _ = cv.parse_verdicts(
        '{"verdicts": [{"index": 0, "action": "retype", "type": "CHẨN_ĐOÁN"}]}', 1
    )
    assert good[0].entity_type == TYPE_DIAGNOSIS


# ------------------------------------------------------------------ deterministic resolution


def test_complete_phrase_beats_a_fragment_of_itself() -> None:
    document = view("chẩn đoán viêm phổi thùy dưới phải")
    fragment = span(document, "viêm phổi", TYPE_DIAGNOSIS, sources={SOURCE_E3})
    complete = span(document, "viêm phổi thùy dưới", TYPE_DIAGNOSIS, sources={SOURCE_QWEN})
    resolution = cr.resolve(document, [fragment, complete])
    assert [e.text for e in resolution.entities] == ["viêm phổi thùy dưới"]
    assert resolution.counters[cr.DROP_OVERLAP] == 1


def test_resolution_is_independent_of_proposal_order() -> None:
    document = view("chẩn đoán viêm phổi thùy dưới phải")
    a = span(document, "viêm phổi", TYPE_DIAGNOSIS, sources={SOURCE_E3})
    b = span(document, "viêm phổi thùy dưới", TYPE_DIAGNOSIS, sources={SOURCE_QWEN})
    assert [e.span for e in cr.resolve(document, [a, b]).entities] == [
        e.span for e in cr.resolve(document, [b, a]).entities
    ]


def test_repeated_exact_mentions_both_survive_resolution() -> None:
    document = view("ho khan nhiều\nvẫn còn ho khan")
    first = span(document, "ho khan", TYPE_SYMPTOM, occurrence=0)
    second = span(document, "ho khan", TYPE_SYMPTOM, occurrence=1)
    entities = cr.resolve(document, [first, second]).entities
    assert len(entities) == 2
    assert entities[0].start != entities[1].start


def test_lab_result_without_a_named_test_is_rejected() -> None:
    document = view("Bệnh nhân 45 tuổi, vào viện ngày 12")
    orphan = span(document, "45", TYPE_RESULT)
    resolution = cr.resolve(document, [orphan])
    assert resolution.entities == []
    assert resolution.counters[cr.DROP_UNANCHORED_RESULT] == 1


def test_lab_result_anchored_to_a_named_test_survives() -> None:
    document = view("Xét nghiệm glucose 7.2 mmol/L")
    name = span(document, "glucose", TYPE_TEST)
    result = span(document, "7.2 mmol/L", TYPE_RESULT)
    entities = cr.resolve(document, [name, result]).entities
    assert {e.type for e in entities} == {TYPE_TEST, TYPE_RESULT}


def test_agreement_outranks_a_single_unsupported_proposal() -> None:
    document = view("bệnh nhân sốt cao liên tục")
    agreed = span(document, "sốt cao", TYPE_SYMPTOM, sources={SOURCE_E3, SOURCE_QWEN})
    lone = Proposal(
        start=agreed.start, end=agreed.end + 9, text="sốt cao liên tục",
        type=TYPE_SYMPTOM, sources=frozenset({SOURCE_ALIAS}),
    )
    lone = dataclasses.replace(
        lone, text=document.source[lone.start : lone.end]
    )
    kept = cr.resolve(document, [agreed, lone]).entities
    assert len(kept) == 1


def test_unsupported_assertion_fields_are_never_emitted_on_lab_types() -> None:
    document = view("Xét nghiệm glucose 7.2 mmol/L")
    name = span(document, "glucose", TYPE_TEST)
    row = cr.organizer_entity(name, ())
    assert set(row) == {"position", "text", "type"}
    assert "assertions" not in row and "candidates" not in row

    symptom = span(view("bệnh nhân sốt"), "sốt", TYPE_SYMPTOM)
    assert cr.organizer_entity(symptom, ("isNegated",))["assertions"] == ["isNegated"]


# ------------------------------------------------------------------ boundary lattice


def test_the_lattice_offers_only_spans_the_document_contains() -> None:
    document = view("chẩn đoán viêm phổi thùy dưới phải")
    proposals = [
        span(document, "viêm phổi", TYPE_DIAGNOSIS),
        span(document, "thùy dưới", TYPE_DIAGNOSIS, sources={SOURCE_QWEN}),
    ]
    lattice = cl.build_lattice(document, proposals)
    for group in lattice.groups:
        for option in group.options:
            assert document.source[option.start : option.end] == option.text


def test_no_expansion_happens_across_a_clause_break() -> None:
    document = view("viêm phổi, tăng huyết áp")
    proposals = [
        span(document, "viêm phổi", TYPE_DIAGNOSIS),
        span(document, "tăng huyết áp", TYPE_DIAGNOSIS),
    ]
    lattice = cl.build_lattice(document, proposals)
    assert lattice.bridges_offered == 0
    assert all(
        option.how != cl.BRIDGE_SOURCE
        for group in lattice.groups for option in group.options
    )


# ------------------------------------------------------------------ assertions


def test_first_person_narration_is_not_family_history() -> None:
    document = view("Tôi bị đau đầu nhiều năm nay")
    entity = span(document, "đau đầu", TYPE_SYMPTOM)
    assert cs.FAMILY not in cs.assert_entity(document, entity, cs.detect_sections(document))


def test_a_family_reference_in_the_clause_marks_family() -> None:
    document = view("Mẹ bệnh nhân bị đái tháo đường")
    entity = span(document, "đái tháo đường", TYPE_DIAGNOSIS)
    assert cs.FAMILY in cs.assert_entity(document, entity, cs.detect_sections(document))


def test_negation_scopes_to_the_entity_clause() -> None:
    document = view("Không sốt, có ho khan")
    sections = cs.detect_sections(document)
    negated = span(document, "sốt", TYPE_SYMPTOM)
    not_negated = span(document, "ho khan", TYPE_SYMPTOM)
    assert cs.NEGATED in cs.assert_entity(document, negated, sections)
    assert cs.NEGATED not in cs.assert_entity(document, not_negated, sections)


def test_history_section_marks_historical() -> None:
    document = view("Tiền sử: viêm phổi\nKhám: sốt cao")
    sections = cs.detect_sections(document)
    assert [s.heading for s in sections] == ["Tiền sử", "Khám"]
    historical = span(document, "viêm phổi", TYPE_DIAGNOSIS)
    current = span(document, "sốt cao", TYPE_SYMPTOM)
    assert cs.HISTORICAL in cs.assert_entity(document, historical, sections)
    assert cs.HISTORICAL not in cs.assert_entity(document, current, sections)


def test_chronicity_alone_does_not_imply_historical() -> None:
    document = view("Bệnh nhân bị viêm phế quản mạn tính")
    entity = span(document, "viêm phế quản", TYPE_DIAGNOSIS)
    assert cs.HISTORICAL not in cs.assert_entity(
        document, entity, cs.detect_sections(document)
    )


def test_an_explicit_past_marker_marks_historical_without_a_section() -> None:
    document = view("Bệnh nhân đã từng mổ ruột thừa")
    entity = span(document, "mổ ruột thừa", TYPE_DIAGNOSIS)
    assert cs.HISTORICAL in cs.assert_entity(document, entity, cs.detect_sections(document))


def test_assertions_are_only_computed_for_types_that_carry_them() -> None:
    document = view("Không thấy glucose 7.2 mmol/L")
    result = span(document, "7.2 mmol/L", TYPE_RESULT)
    assert cs.assert_entity(document, result, cs.detect_sections(document)) == ()
    outcome = cs.run_assertions(document, [result])
    assert outcome.assertions == {}


# ------------------------------------------------------------------ end to end, injected fakes


class FakeQwen:
    """Answers the proposer prompt and every verifier prompt from canned replies."""

    def __init__(self, proposal_reply: str, verdict_reply: str = "") -> None:
        self.proposal_reply = proposal_reply
        self.verdict_reply = verdict_reply
        self.prompts: list[str] = []

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "CANDIDATES:" in prompt:
            if self.verdict_reply:
                return self.verdict_reply
            count = prompt.count("\n[")+ (1 if prompt.count("[0]") else 0)
            return json.dumps({
                "verdicts": [
                    {"index": i, "action": "accept"} for i in range(max(count, 1))
                ]
            })
        return self.proposal_reply

    def unload(self) -> None:
        return None


NOTE = "Tiền sử: tăng huyết áp\nKhám: bệnh nhân sốt cao, ho khan\nXử trí: paracetamol 500mg"


def seed_entity(text: str, entity_type: str) -> dict[str, Any]:
    start = NOTE.index(text)
    return {"text": text, "type": entity_type, "position": [start, start + len(text)],
            "assertions": []}


def test_end_to_end_pass_produces_aligned_entities_and_diagnostics() -> None:
    qwen = FakeQwen(json.dumps({"entities": [
        {"line_id": "L002", "text": "ho khan", "type": TYPE_SYMPTOM, "occurrence": 0},
        {"line_id": "L003", "text": "paracetamol 500mg", "type": TYPE_DRUG, "occurrence": 0},
        {"line_id": "L003", "text": "khong co trong van ban", "type": TYPE_DRUG},
    ]}))
    outcome = crt.process_document(
        "1", NOTE, [seed_entity("sốt cao", TYPE_SYMPTOM)], qwen=qwen,
    )
    texts = {e.text for e in outcome.entities}
    assert "sốt cao" in texts and "ho khan" in texts and "paracetamol 500mg" in texts
    for entity in outcome.entities:
        assert NOTE[entity.start : entity.end] == entity.text
    assert outcome.counters["proposals_e3"] == 1
    assert outcome.counters["proposals_qwen_context"] == 2   # the third did not align
    assert outcome.counters[cd.REJECT_NOT_IN_SOURCE] == 1


def test_one_pass_derives_three_ablation_variants() -> None:
    qwen = FakeQwen(json.dumps({"entities": [
        {"line_id": "L002", "text": "ho khan", "type": TYPE_SYMPTOM, "occurrence": 0},
    ]}))
    outcome = crt.process_document(
        "1", NOTE, [seed_entity("sốt cao", TYPE_SYMPTOM)], qwen=qwen,
    )
    asked = len(qwen.prompts)
    control = crt.variant_entities(outcome, crt.VARIANT_CONTROL)
    high = crt.variant_entities(outcome, crt.VARIANT_HIGH)
    broad = crt.variant_entities(outcome, crt.VARIANT_BROAD)

    assert {e.text for e in control} == {"sốt cao"}          # E3 reproduced exactly
    assert len(broad) >= len(high) >= len(control)
    assert "ho khan" in {e.text for e in broad}
    assert len(qwen.prompts) == asked                        # variants derived, not re-run


def test_the_control_variant_reproduces_the_seed_exactly() -> None:
    seeds = [seed_entity("sốt cao", TYPE_SYMPTOM), seed_entity("ho khan", TYPE_SYMPTOM)]
    outcome = crt.process_document("1", NOTE, seeds, qwen=None)
    control = crt.variant_entities(outcome, crt.VARIANT_CONTROL)
    assert {(e.start, e.end) for e in control} == {
        (s["position"][0], s["position"][1]) for s in seeds
    }


def test_serialized_output_carries_no_unsupported_field(tmp_path: Path) -> None:
    qwen = FakeQwen(json.dumps({"entities": [
        {"line_id": "L003", "text": "paracetamol 500mg", "type": TYPE_DRUG},
    ]}))
    outcome = crt.process_document(
        "1", NOTE, [seed_entity("sốt cao", TYPE_SYMPTOM)], qwen=qwen,
    )
    counts = crt.write_variants(tmp_path, [outcome])
    assert set(counts) == set(crt.VARIANTS)

    rows = json.loads(
        (tmp_path / f"entities_{crt.VARIANT_BROAD}" / "1.json").read_text(encoding="utf-8")
    )
    allowed = {"position", "text", "type", "assertions"}
    for row in rows:
        assert set(row) <= allowed
        assert NOTE[row["position"][0] : row["position"][1]] == row["text"]
        if row["type"] in {TYPE_TEST, TYPE_RESULT}:
            assert "assertions" not in row
    assert (tmp_path / "proposal-records.jsonl").is_file()


def test_diagnostics_attribute_every_entity_to_its_proposal_sources() -> None:
    qwen = FakeQwen(json.dumps({"entities": [
        {"line_id": "L002", "text": "ho khan", "type": TYPE_SYMPTOM},
        {"line_id": "L002", "text": "sốt cao", "type": TYPE_SYMPTOM},
    ]}))
    outcome = crt.process_document(
        "1", NOTE, [seed_entity("sốt cao", TYPE_SYMPTOM)], qwen=qwen,
    )
    summary = crt.diagnostics([outcome], {"e3_control": 1})
    buckets = summary["final_entities_by_proposal_source"]
    assert buckets["agreement"] >= 1        # sốt cao found by both E3 and Qwen
    assert buckets["qwen_only"] >= 1        # ho khan found by Qwen alone
    assert summary["counters"]["lattice_groups"] >= 1


def test_the_verifier_can_be_disabled_without_changing_the_control() -> None:
    config = crt.ContextualConfig(use_verifier=False, use_qwen_proposer=False)
    outcome = crt.process_document(
        "1", NOTE, [seed_entity("sốt cao", TYPE_SYMPTOM)], qwen=None, config=config
    )
    assert [e.text for e in outcome.entities] == ["sốt cao"]


# ------------------------------------------------------------------ GraphCENT is untouched


def test_graphcent_still_takes_indices_and_never_a_model_supplied_id() -> None:
    from mednorm_vi.graphcent.disambiguation import RUBRIC, parse_decision

    offered = ["J189", "J180"]
    invented = parse_decision('{"decision":"SELECT","selected_indices":[],'
                              '"candidate_ids":["J999"]}', offered)
    assert invented.decision == "NULL"

    good = parse_decision(
        '{"decision":"SELECT","selected_indices":[0],"reason_codes":[]}', offered
    )
    assert good.selected_indices == (0,)
    assert "You cannot write a medical code" in RUBRIC


def test_graphcent_rubric_now_refuses_unsupported_specificity() -> None:
    from mednorm_vi.graphcent.disambiguation import RUBRIC

    for taught in (
        "SEMANTIC EQUIVALENCE", "anatomical site", "pregnancy", "malignancy",
        "acuity or chronicity", "dosage, strength, concentration, form and route",
        "more specific child concept", "alternative granularities",
    ):
        assert taught in RUBRIC, taught
    assert "DEFAULT ANSWER IS NULL" in RUBRIC


def test_0081_deploys_no_additional_base_model() -> None:
    """The 0081 CLI reaches for the canonical Qwen, never a new checkpoint."""
    source = CLI_PATH.read_text(encoding="utf-8")
    assert "QwenDisambiguator" in source
    assert "from_pretrained" not in source
    assert "snapshot_download" not in source
    for banned in ("SapBERT", "BioBERT", "ClinLinker", "AutoModel"):
        assert banned not in source, banned


def test_the_canonical_budget_is_reused_not_recomputed(tmp_path: Path) -> None:
    from mednorm_vi.graphcent.acquisition import MANIFEST_NAME
    from mednorm_vi.graphcent.models import PARAMETER_CAP

    root = tmp_path / "models"
    root.mkdir()
    (root / MANIFEST_NAME).write_text(json.dumps({
        "deployed": [
            {"name": "Qwen/Qwen3-8B", "parameter_count": 8_190_735_360,
             "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
        ],
        "total_deployed_parameters": 8_190_735_360,
    }), encoding="utf-8")
    args = cli.build_parser().parse_args(["budget", "--model-root", str(root)])
    assert args.func(args) == 0
    assert 8_190_735_360 < PARAMETER_CAP

    (root / MANIFEST_NAME).write_text(json.dumps({
        "deployed": [], "total_deployed_parameters": PARAMETER_CAP,
    }), encoding="utf-8")
    assert args.func(args) == 3


def test_no_stub_markers_in_the_0081_production_path() -> None:
    import re

    banned = re.compile(r"NotImplementedError|TODO|FIXME|implement on Colab|placeholder")
    paths = [
        *sorted((ROOT / "src/mednorm_vi/contextual").glob("*.py")),
        CLI_PATH,
    ]
    for path in paths:
        assert not banned.search(path.read_text(encoding="utf-8")), path.name


def test_the_profile_declares_the_proposal_sources_and_the_shared_model() -> None:
    import yaml

    config = yaml.safe_load(
        (ROOT / "configs/pipeline/contextual_0081.yaml").read_text(encoding="utf-8")
    )
    assert config["use_qwen_proposer"] is True
    assert config["use_alias_proposer"] is True
    assert config["use_verifier"] is True
    assert config["qwen_model_root"]
    specs = cli.build_parser().parse_args(
        ["smoke", "--input-dir", "x", "--run-dir", "y"]
    )
    assert specs.func is cli.cmd_smoke


@pytest.mark.parametrize("variant", crt.VARIANTS)
def test_every_variant_serializes_valid_organizer_rows(variant: str) -> None:
    qwen = FakeQwen(json.dumps({"entities": [
        {"line_id": "L001", "text": "tăng huyết áp", "type": TYPE_DIAGNOSIS},
    ]}))
    outcome = crt.process_document(
        "1", NOTE, [seed_entity("sốt cao", TYPE_SYMPTOM)], qwen=qwen,
    )
    for row in crt.serialize(outcome, variant):
        assert row["type"] in {
            TYPE_SYMPTOM, TYPE_DIAGNOSIS, TYPE_TEST, TYPE_RESULT, TYPE_DRUG
        }
        assert NOTE[row["position"][0] : row["position"][1]] == row["text"]
        assert "candidates" not in row      # candidates remain GraphCENT's decision


# ================== repairs for the first real Colab smoke (documents 1, 10, 100) ==========
#
# Observed: proposer_parse_failed=3 (every document), verifier_invalid_index=675 against 357
# offered options, 75 alias-only junk entities, contextual_high identical to e3_control, and
# 'Thiếu men G6PD' fragmented into 'Thiếu men' + 'G6PD'.


class RecordingQwen:
    """Answers each specialized pass separately and records the generation budget it saw."""

    def __init__(self, replies: dict[str, str] | None = None, verdict: str = "",
                 max_new_tokens: int = 1536) -> None:
        self.replies = replies or {}
        self.verdict = verdict
        self.max_new_tokens = max_new_tokens
        self.prompts: list[str] = []

    def _pass_of(self, prompt: str) -> str:
        for pass_name, rubric in cp.RUBRICS.items():
            if rubric.split("\n", 1)[0] in prompt:
                return pass_name
        return ""

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "CANDIDATES:" in prompt:
            return self.verdict or json.dumps({"verdicts": [{"index": 0, "action": "accept"}]})
        return self.replies.get(self._pass_of(prompt), '{"entities": []}')

    def ask_detailed(self, prompt: str) -> tuple[str, dict[str, Any]]:
        reply = self.ask(prompt)
        tokens = max(len(reply) // 4, 1)
        return reply, {
            "generated_tokens": tokens,
            "hit_token_limit": tokens >= self.max_new_tokens,
            "max_new_tokens": self.max_new_tokens,
        }


def entities_payload(rows: list[tuple[str, str, str]]) -> str:
    return json.dumps({
        "entities": [
            {"line_id": line, "text": text, "type": kind, "occurrence": 0}
            for line, text, kind in rows
        ]
    })


# ---------------------------------------------------- root cause 1: truncation and parsing


def test_a_truncated_reply_is_reported_as_truncation_not_as_silence() -> None:
    """The exact first-smoke failure: generation ran out of budget mid-JSON."""
    cut_off = '{"entities": [{"line_id": "L001", "text": "sốt", "type": "TRIỆU_CHỨNG"'
    result = cp.parse_pass(
        cp.PASS_DIAGNOSIS, cut_off, generated_tokens=1536, hit_token_limit=True
    )
    assert result.proposals == []
    assert result.report.category == cp.PARSE_TRUNCATED
    assert result.report.hit_token_limit is True
    assert result.report.json_extracted is False
    assert result.report.characters == len(cut_off)


def test_generation_diagnostics_record_shape_and_never_reasoning() -> None:
    reply = (
        "<think>the patient seems to have a fever, I should consider...</think>\n"
        '```json\n{"entities": [{"line_id": "L001", "text": "sốt", '
        '"type": "TRIỆU_CHỨNG", "occurrence": 0}]}\n```'
    )
    result = cp.parse_pass(cp.PASS_DIAGNOSIS, reply, generated_tokens=90)
    assert [p.text for p in result.proposals] == ["sốt"]

    report = result.report.as_dict()
    for value in (report["excerpt_head"], report["excerpt_tail"]):
        assert "<think>" not in value and "I should consider" not in value
    assert len(report["excerpt_head"]) <= cp.DEBUG_EXCERPT
    assert report["json_extracted"] is True
    assert report["generated_tokens"] == 90
    assert report["rows_kept"] == 1


def test_qwen3_thinking_blocks_and_markdown_fences_are_handled() -> None:
    for reply in (
        '```json\n{"entities": []}\n```',
        '<think>reasoning</think>{"entities": []}',
        'Here is the result:\n{"entities": []}\nLet me know if you need more.',
        '<think>unterminated reasoning {"entities": [] }',
    ):
        result = cp.parse_pass(cp.PASS_LAB, reply)
        # an unterminated thinking block leaves nothing to parse, which is reported as
        # empty rather than guessed at
        assert result.report.category in {cp.PARSE_OK, cp.PARSE_NO_JSON, cp.PARSE_EMPTY}


def test_json_extraction_is_brace_matched_not_greedy() -> None:
    body, reason = cp.extract_json_object('{"entities": []} trailing {"other": 1}')
    assert reason == cp.PARSE_OK
    assert body == '{"entities": []}'          # not spanning to the last brace

    body, reason = cp.extract_json_object('{"entities": [{"text": "a"')
    assert body == "" and reason == cp.PARSE_TRUNCATED

    body, reason = cp.extract_json_object("no object at all")
    assert body == "" and reason == cp.PARSE_NO_JSON

    # a brace inside a string must not close the object
    body, reason = cp.extract_json_object('{"text": "a } b", "n": 1}')
    assert reason == cp.PARSE_OK and body.endswith("}")


def test_the_structured_extraction_budget_is_not_the_disambiguation_default() -> None:
    """256 tokens is a one-word answer's budget; it is what truncated every document."""
    import yaml

    profile = yaml.safe_load(
        (ROOT / "configs/pipeline/contextual_0081.yaml").read_text(encoding="utf-8")
    )
    assert profile["max_new_tokens"] >= 1024
    source = CLI_PATH.read_text(encoding="utf-8")
    assert "max_new_tokens=int(" in source.replace(" ", "")


# ---------------------------------------------------- root cause 2: three separate passes


def test_each_pass_only_accepts_the_types_it_asked_for() -> None:
    leaked = cp.parse_pass(
        cp.PASS_MEDICATION,
        entities_payload([("L001", "sốt", TYPE_SYMPTOM), ("L001", "aspirin", TYPE_DRUG)]),
    )
    assert [p.text for p in leaked.proposals] == ["aspirin"]
    assert leaked.counters[cp.PARSE_WRONG_TYPE_FOR_PASS] == 1


def test_one_failing_pass_does_not_discard_the_other_two() -> None:
    note = "Khám: sốt cao\nThuốc: aspirin 325mg\nXét nghiệm: glucose 7.2 mmol/L"
    qwen = RecordingQwen({
        cp.PASS_DIAGNOSIS: "the model rambled instead of answering",   # this pass fails
        cp.PASS_MEDICATION: entities_payload([("L002", "aspirin 325mg", TYPE_DRUG)]),
        cp.PASS_LAB: entities_payload([("L003", "glucose", TYPE_TEST)]),
    })
    outcome = crt.process_document("1", note, [], qwen=qwen)
    texts = {e.text for e in outcome.entities}
    assert "aspirin 325mg" in texts and "glucose" in texts
    assert outcome.counters["proposer_passes_failed"] == 1
    assert outcome.counters["proposer_passes_parsed"] == 2


def test_all_three_passes_are_asked_exactly_once_per_document() -> None:
    qwen = RecordingQwen()
    crt.process_document("1", "Khám: sốt cao", [], qwen=qwen)
    asked = [qwen._pass_of(p) for p in qwen.prompts if "CANDIDATES:" not in p]
    assert sorted(asked) == sorted(cp.PASSES)


# ---------------------------------------------------- root cause 3: verifier index space


def test_the_schema_example_never_shows_an_index_that_was_not_offered() -> None:
    """This is the 675-invalid-index bug: a fixed 0/1/2 example against one-option groups."""
    single = cv.schema_for(1)
    assert '"index": 1' not in single and '"index": 2' not in single
    assert "ONLY valid index values are 0" in single

    three = cv.schema_for(3)
    assert '"index": 3' not in three
    assert "0 to 2" in three


def test_a_one_option_group_teaches_only_index_zero() -> None:
    document = view("bệnh nhân viêm phổi")
    group = cl.LatticeGroup(
        start=10, end=19, options=(span(document, "viêm phổi", TYPE_DIAGNOSIS),)
    )
    prompt = cv.build_verification_prompt(document, group)
    assert "[0]" in prompt and "[1]" not in prompt
    assert "1 candidate(s)" in prompt

    # the model copying the old fixed example is still refused, but nothing invites it now
    verdicts, counters = cv.parse_verdicts(
        '{"verdicts": [{"index": 0, "action": "accept"}, {"index": 1, "action": "reject"}, '
        '{"index": 2, "action": "reject"}]}', 1
    )
    assert [v.index for v in verdicts] == [0]
    assert counters[cv.INVALID_INDEX] == 2


def test_indices_are_local_to_one_invocation_and_never_global_ids() -> None:
    document = view("sốt cao và ho khan nhiều")
    first = cl.LatticeGroup(start=0, end=7, options=(span(document, "sốt cao", TYPE_SYMPTOM),))
    second = cl.LatticeGroup(
        start=11, end=18, options=(span(document, "ho khan", TYPE_SYMPTOM),)
    )
    for group in (first, second):
        prompt = cv.build_verification_prompt(document, group)
        assert "[0]" in prompt                      # both start at zero: one index space
        verdicts, counters = cv.parse_verdicts(
            '{"verdicts": [{"index": 0, "action": "accept"}]}', len(group.options)
        )
        assert [v.index for v in verdicts] == [0] and not counters


@pytest.mark.parametrize("bad", ["-1", "7", '"0"', "null", "1.5"])
def test_a_foreign_or_malformed_index_fails_closed(bad: str) -> None:
    verdicts, counters = cv.parse_verdicts(
        '{"verdicts": [{"index": ' + bad + ', "action": "accept"}]}', 2
    )
    assert verdicts == []
    assert sum(counters.values()) >= 1


def test_verifier_parsing_survives_fences_and_thinking() -> None:
    verdicts, counters = cv.parse_verdicts(
        '<think>which one</think>```json\n{"verdicts": [{"index": 0, "action": "accept"}]}\n```',
        1,
    )
    assert [v.index for v in verdicts] == [0] and not counters


# ---------------------------------------------------- root cause 4: alias flooding


def test_a_form_naming_many_governed_concepts_is_refused() -> None:
    class Ambiguous:
        records = {
            f"C{i}": {"canonical_name": "khác", "aliases": []} for i in range(9)
        } | {"D1": {"canonical_name": "viêm phổi thùy dưới", "aliases": []}}

    lexicon = ca.build_lexicon([(Ambiguous(), TYPE_DIAGNOSIS)])
    assert lexicon.skipped[ca.REJECT_AMBIGUOUS_FORM] >= 1
    assert ca.find_alias_hits("bệnh khác", lexicon) == []
    assert ca.find_alias_hits("viêm phổi thùy dưới", lexicon)


def test_a_token_common_across_labels_is_not_an_identifying_alias() -> None:
    class Common:
        # "bệnh" appears in most labels, so it is a word of the vocabulary, not a term
        records = {
            f"C{i}": {"canonical_name": f"bệnh loại {i}", "aliases": []} for i in range(10)
        } | {"C99": {"canonical_name": "bệnh", "aliases": []}}

    lexicon = ca.build_lexicon([(Common(), TYPE_DIAGNOSIS)])
    assert lexicon.skipped.get(ca.REJECT_COMMON_TOKEN, 0) >= 1
    assert [h.text for h in ca.find_alias_hits("bệnh nhân có bệnh", lexicon)] == []


def test_a_governed_label_marked_suspect_is_not_used_as_an_alias() -> None:
    class Flagged:
        records = {
            "A1": {"canonical_name": "viêm phổi thùy dưới",
                   "metadata": {"name_quality": "clean", "quality_flags": ""}},
            "A2": {"canonical_name": "rối loạn tâm thần và",
                   "metadata": {"name_quality": "suspect",
                                "quality_flags": "suspect_trailing_function_word"}},
        }

    lexicon = ca.build_lexicon([(Flagged(), TYPE_DIAGNOSIS)])
    assert lexicon.skipped[ca.REJECT_SUSPECT_LABEL] == 1
    assert ca.find_alias_hits("rối loạn tâm thần và mệt", lexicon) == []


def test_ambiguity_is_keyed_on_tokens_so_punctuation_cannot_disguise_a_form() -> None:
    assert ca.form_key("Bệnh:") == ca.form_key("bệnh") == "bệnh"
    assert ca.form_key("khác.") == "khác"


def test_an_alias_only_span_never_becomes_a_final_entity() -> None:
    """The 75 alias-only junk entities: an alias is evidence, not an entity."""
    note = "Bệnh nhân không có bệnh khác, thể trạng ổn."
    lexicon = ca.AliasLexicon()
    for word in ("bệnh", "khác", "thể"):
        lexicon.add(word, TYPE_DIAGNOSIS)

    outcome = crt.process_document("1", note, [], qwen=None, lexicon=lexicon)
    assert outcome.entities == []
    assert outcome.counters["alias_hits"] >= 3
    assert outcome.counters.get("alias_only_group_skipped", 0) >= 1


def test_an_alias_still_strengthens_a_span_another_source_found() -> None:
    note = "Chẩn đoán: viêm phổi thùy dưới"
    lexicon = ca.AliasLexicon()
    lexicon.add("viêm phổi thùy dưới", TYPE_DIAGNOSIS)
    qwen = RecordingQwen({
        cp.PASS_DIAGNOSIS: entities_payload(
            [("L001", "viêm phổi thùy dưới", TYPE_DIAGNOSIS)]
        ),
    })
    outcome = crt.process_document("1", note, [], qwen=qwen, lexicon=lexicon)
    entity = next(e for e in outcome.entities if e.text == "viêm phổi thùy dưới")
    assert entity.sources == {SOURCE_QWEN, SOURCE_ALIAS}
    assert crt.high_confidence(entity) == (True, "qwen_with_governed_alias_support")


# ---------------------------------------------------- root cause 5: contextual_high


def test_contextual_high_can_contain_a_qwen_addition_e3_never_found() -> None:
    """`contextual_high == e3_control` meant 0081 could never repair E3 recall.

    High confidence now asks what INDEPENDENT evidence a mention has, not whether E3 found
    it - otherwise a mention E3 missed could never qualify, and E3 recall is the thing 0081
    exists to fix.
    """
    note = "Khám: sốt cao. Chẩn đoán: viêm phổi thùy dưới"
    seed = [{"text": "sốt cao", "type": TYPE_SYMPTOM,
             "position": [note.index("sốt cao"), note.index("sốt cao") + 7],
             "assertions": []}]
    lexicon = ca.AliasLexicon()
    lexicon.add("viêm phổi thùy dưới", TYPE_DIAGNOSIS)
    qwen = RecordingQwen({
        cp.PASS_DIAGNOSIS: entities_payload([
            ("L001", "viêm phổi thùy dưới", TYPE_DIAGNOSIS),
        ]),
    })
    outcome = crt.process_document("1", note, seed, qwen=qwen, lexicon=lexicon)

    control = {e.text for e in crt.variant_entities(outcome, crt.VARIANT_CONTROL)}
    high = {e.text for e in crt.variant_entities(outcome, crt.VARIANT_HIGH)}
    assert control == {"sốt cao"}
    assert "viêm phổi thùy dưới" not in control
    assert high > control, "contextual_high must be able to add contextual mentions"
    assert "viêm phổi thùy dưới" in high


def test_a_qwen_span_completing_an_e3_fragment_is_high_confidence() -> None:
    note = "Chẩn đoán: Thiếu men G6PD bẩm sinh"
    fragment = note.index("G6PD")
    seed = [{"text": "G6PD", "type": TYPE_DIAGNOSIS,
             "position": [fragment, fragment + 4], "assertions": []}]
    qwen = RecordingQwen({
        cp.PASS_DIAGNOSIS: entities_payload([("L001", "Thiếu men G6PD", TYPE_DIAGNOSIS)]),
    })
    outcome = crt.process_document("1", note, seed, qwen=qwen)
    complete = next(e for e in outcome.entities if e.text == "Thiếu men G6PD")
    assert complete.subsumes_seed
    assert crt.high_confidence(complete) == (True, "qwen_completed_an_e3_fragment")
    assert complete.text in {
        e.text for e in crt.variant_entities(outcome, crt.VARIANT_HIGH)
    }


def test_an_unsupported_qwen_span_reaches_broad_but_not_high() -> None:
    note = "Khám: bệnh nhân ho khan"
    qwen = RecordingQwen({
        cp.PASS_DIAGNOSIS: entities_payload([("L001", "ho khan", TYPE_SYMPTOM)]),
    })
    outcome = crt.process_document("1", note, [], qwen=qwen)
    high = {e.text for e in crt.variant_entities(outcome, crt.VARIANT_HIGH)}
    broad = {e.text for e in crt.variant_entities(outcome, crt.VARIANT_BROAD)}
    assert "ho khan" in broad and "ho khan" not in high


# ---------------------------------------------------- root cause 6: complete concepts


def test_the_complete_clinical_phrase_beats_its_own_fragments() -> None:
    """`Thiếu men` + `G6PD` must be resolvable to `Thiếu men G6PD`.

    The phrase is a regression example only; nothing about it appears in production logic.
    """
    note = "Chẩn đoán: Thiếu men G6PD bẩm sinh"
    complete = span(note_view(note), "Thiếu men G6PD", TYPE_DIAGNOSIS,
                    sources={SOURCE_QWEN})
    left = span(note_view(note), "Thiếu men", TYPE_DIAGNOSIS, sources={SOURCE_E3})
    right = span(note_view(note), "G6PD", TYPE_DIAGNOSIS, sources={SOURCE_E3})

    resolution = cr.resolve(note_view(note), [left, right, complete])
    assert [e.text for e in resolution.entities] == ["Thiếu men G6PD"]


def note_view(source: str) -> cd.DocumentView:
    return view(source)


def test_a_medication_keeps_its_whole_name_and_regimen() -> None:
    note = "Thuốc: metoprolol 25mg po bid, aspirin 325mg"
    qwen = RecordingQwen({
        cp.PASS_MEDICATION: entities_payload([
            ("L001", "metoprolol 25mg po bid", TYPE_DRUG),
            ("L001", "aspirin 325mg", TYPE_DRUG),
        ]),
    })
    outcome = crt.process_document("1", note, [], qwen=qwen)
    texts = {e.text for e in outcome.entities}
    assert "metoprolol 25mg po bid" in texts and "aspirin 325mg" in texts
    assert "metoprolol" not in texts        # no fragment of a longer name survives


def test_repeated_occurrences_at_distinct_offsets_both_survive() -> None:
    note = "Khám: ho khan\nTheo dõi: ho khan"
    qwen = RecordingQwen({
        cp.PASS_DIAGNOSIS: json.dumps({"entities": [
            {"line_id": "L001", "text": "ho khan", "type": TYPE_SYMPTOM, "occurrence": 0},
            {"line_id": "L002", "text": "ho khan", "type": TYPE_SYMPTOM, "occurrence": 0},
        ]}),
    })
    outcome = crt.process_document("1", note, [], qwen=qwen)
    starts = sorted(e.start for e in outcome.entities if e.text == "ho khan")
    assert len(starts) == 2 and starts[0] != starts[1]


def test_no_final_entity_is_a_mid_token_substring() -> None:
    note = "Bệnh nhân bị hen suyễn ASTma nặng"
    lexicon = ca.AliasLexicon()
    lexicon.add("AST", TYPE_TEST)
    outcome = crt.process_document("1", note, [], qwen=None, lexicon=lexicon)
    assert outcome.entities == []


# ---------------------------------------------------- root cause 7: assertions


def test_the_observed_bad_family_assertions_no_longer_fire() -> None:
    for text, mention in (
        ("Không có tiền sử gì của bố mẹ", "Không"),
        ("Bà kể các triệu chứng kéo dài", "các triệu chứng"),
    ):
        document = view(text)
        entity = span(document, mention, TYPE_SYMPTOM)
        assert cs.FAMILY not in cs.assert_entity(
            document, entity, cs.detect_sections(document)
        )


def test_a_kinship_word_after_the_entity_does_not_make_it_family() -> None:
    document = view("Đau đầu, sau đó mẹ cũng bị")
    entity = span(document, "Đau đầu", TYPE_SYMPTOM)
    assert cs.FAMILY not in cs.assert_entity(document, entity, cs.detect_sections(document))


def test_an_address_pronoun_counts_as_family_only_inside_a_family_section() -> None:
    plain = view("Bà bị tăng huyết áp")
    entity = span(plain, "tăng huyết áp", TYPE_DIAGNOSIS)
    assert cs.FAMILY not in cs.assert_entity(plain, entity, cs.detect_sections(plain))

    declared = view("Tiền sử gia đình: bà bị tăng huyết áp")
    entity = span(declared, "tăng huyết áp", TYPE_DIAGNOSIS)
    assert cs.FAMILY in cs.assert_entity(declared, entity, cs.detect_sections(declared))


# ---------------------------------------------------- root cause 8: the control


def test_e3_control_reproduces_the_seed_whatever_qwen_and_the_aliases_do() -> None:
    note = "Khám: sốt cao, ho khan. Thuốc: aspirin 325mg"
    seed = [
        {"text": "sốt cao", "type": TYPE_SYMPTOM,
         "position": [note.index("sốt cao"), note.index("sốt cao") + 7], "assertions": []},
        {"text": "aspirin 325mg", "type": TYPE_DRUG,
         "position": [note.index("aspirin 325mg"), note.index("aspirin 325mg") + 13],
         "assertions": []},
    ]
    lexicon = ca.AliasLexicon()
    for word in ("bệnh", "khác"):
        lexicon.add(word, TYPE_DIAGNOSIS)
    qwen = RecordingQwen({
        cp.PASS_DIAGNOSIS: entities_payload([("L001", "ho khan", TYPE_SYMPTOM)]),
        cp.PASS_MEDICATION: entities_payload([("L001", "aspirin 325mg", TYPE_DRUG)]),
    })
    outcome = crt.process_document("1", note, seed, qwen=qwen, lexicon=lexicon)

    control = crt.variant_entities(outcome, crt.VARIANT_CONTROL)
    assert {(e.start, e.end, e.type) for e in control} == {
        (s["position"][0], s["position"][1], s["type"]) for s in seed
    }
    rows = crt.serialize(outcome, crt.VARIANT_CONTROL)
    assert [r["text"] for r in rows] == ["sốt cao", "aspirin 325mg"]
    for row in rows:
        assert set(row) <= {"position", "text", "type", "assertions"}


def test_variants_never_disagree_about_a_shared_entity() -> None:
    note = "Khám: sốt cao, ho khan"
    seed = [{"text": "sốt cao", "type": TYPE_SYMPTOM,
             "position": [note.index("sốt cao"), note.index("sốt cao") + 7],
             "assertions": []}]
    qwen = RecordingQwen({
        cp.PASS_DIAGNOSIS: entities_payload([("L001", "ho khan", TYPE_SYMPTOM)]),
    })
    outcome = crt.process_document("1", note, seed, qwen=qwen)
    by_variant = {v: crt.serialize(outcome, v) for v in crt.VARIANTS}
    shared = {r["text"]: r for r in by_variant[crt.VARIANT_CONTROL]}
    for variant, rows in by_variant.items():
        for row in rows:
            if row["text"] in shared:
                assert row == shared[row["text"]], variant
