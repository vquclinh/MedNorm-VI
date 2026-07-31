"""Milestone 4A: governed decoding profiles, ICD serialization probe, prepared indices.

Three things landed here, and they share one motive: the Audit-0059 baseline produced
three leaderboard artifacts, two of which could not be regenerated from the repository
because an untracked script post-processed the JSON. Everything in this module exists
so that never has to happen again — the profile is config, the derivation is a tracked
tool, and both are tested against the properties that make them safe.

The serialization tests are deliberately paranoid about *reversibility*. The dotted
probe's entire value is that it isolates one variable; if the transform could lose or
alter a code, the leaderboard delta it produces would mean nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mednorm_vi.inference.code_format import (
    MalformedIcdCode,
    classify_icd_code,
    to_dotless,
    to_dotted,
    transform_diagnosis_codes,
)
from mednorm_vi.inference.derive_submission import (
    DERIVE_ICD_DOTTED,
    DERIVE_TRUNCATE,
    derive_submission,
    validate_derivation,
)
from mednorm_vi.kb.competition.topk import (
    COMPETITION_ADAPTIVE,
    COMPETITION_TOP1,
    DEFAULT_DECODING_PROFILE,
    RankedCandidate,
    apply_competition_topk,
    resolve_decoding_profile,
)

REPO = Path(__file__).resolve().parents[2]


# --- ICD serialization --------------------------------------------------------


@pytest.mark.parametrize(
    ("dotless", "dotted"),
    [
        ("I269", "I26.9"),
        ("I509", "I50.9"),
        ("I248", "I24.8"),
        ("I7000", "I70.00"),
        ("K650", "K65.0"),
        ("N03", "N03"),
        ("A00", "A00"),
    ],
)
def test_dotting_matches_the_specified_examples(dotless: str, dotted: str) -> None:
    assert to_dotted(dotless) == dotted


def test_dotting_is_reversible_and_idempotent() -> None:
    """The probe's whole claim rests on this: nothing but a dot may move."""
    for code in ("I269", "I7000", "K650", "N03", "Z00", "M4802"):
        assert to_dotless(to_dotted(code)) == code
        assert to_dotted(to_dotted(code)) == to_dotted(code)


def test_the_accepted_shape_matches_the_active_vietnamese_kb() -> None:
    """The transform's grammar is derived from the KB, not from ICD-10 in general.

    All 15,308 codes in the active competition-v3 ICD index match
    ``[A-Z][0-9]{2}[0-9A-Z]*`` with length 3-5 and no letter after position 3. An
    ICD-10-CM shape such as ``C7A1`` is therefore correctly rejected here: the
    organizer confirmed a *Vietnamese* ICD-10 and `icd_format_hypotheses_v1.yaml`
    explicitly records `assume_who_vs_cm: false`, so silently accepting a CM-only
    shape would be inventing coverage this KB does not have.
    """
    index = REPO / "indices" / "candidate" / "icd10_vi" / "competition-v3" / "index.json"
    if not index.is_file():
        pytest.skip("competition v3 ICD index not built locally")
    codes = [r["concept_id"] for r in json.loads(index.read_text(encoding="utf-8"))["records"]]
    assert codes
    assert all(classify_icd_code(c) == "dotless" for c in codes)
    assert classify_icd_code("C7A1") == "malformed"


def test_a_three_character_category_is_left_alone() -> None:
    assert to_dotted("N03") == "N03"
    assert classify_icd_code("N03") == "dotless"


def test_an_already_dotted_code_is_left_alone() -> None:
    assert to_dotted("I26.9") == "I26.9"
    assert classify_icd_code("I26.9") == "dotted"


@pytest.mark.parametrize("bad", ["", "12", "269I", "I2", "I26.", ".I269", "I26.9.1", "i269"])
def test_a_malformed_code_fails_closed(bad: str) -> None:
    """A probe that silently 'cleaned up' a bad code would introduce a second variable."""
    assert classify_icd_code(bad) == "malformed"
    with pytest.raises(MalformedIcdCode):
        to_dotted(bad)


def test_transform_preserves_order_and_multiplicity() -> None:
    codes = ["I509", "N03", "I269", "I509"]
    out, report = transform_diagnosis_codes(codes)
    assert out == ["I50.9", "N03", "I26.9", "I50.9"]
    assert report.transformed == 3
    assert report.unchanged_category == 1
    assert report.malformed == ()


# --- submission derivation ----------------------------------------------------


def _write_submission(root: Path, docs: dict[int, list[dict[str, object]]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for doc_id, ents in docs.items():
        (root / f"{doc_id}.json").write_text(
            json.dumps(ents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return root


def _sample_docs(n: int = 3) -> dict[int, list[dict[str, object]]]:
    docs: dict[int, list[dict[str, object]]] = {}
    for i in range(1, n + 1):
        docs[i] = [
            {
                "text": "viem phoi",
                "type": "CHẨN_ĐOÁN",
                "candidates": ["J189", "J18"],
                "assertions": ["isNegated"],
                "position": [0, 9],
            },
            {
                "text": "paracetamol",
                "type": "THUỐC",
                "candidates": ["161"],
                "assertions": [],
                "position": [10, 21],
            },
            {"text": "sot", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [22, 25]},
        ]
    return docs


def test_icd_dotted_derivation_changes_only_diagnosis_codes(tmp_path: Path) -> None:
    src = _write_submission(tmp_path / "src", _sample_docs())
    out = tmp_path / "derived" / "output"
    result = derive_submission(
        source_dir=src, output_dir=out, derivation=DERIVE_ICD_DOTTED, expected_documents=3
    )
    assert result.documents == 3
    assert result.code_report is not None
    assert result.code_report.transformed == 3  # one J189 per document
    assert result.code_report.unchanged_category == 3  # one J18 per document

    validation = validate_derivation(source_dir=src, derived_dir=out, derivation=DERIVE_ICD_DOTTED)
    assert validation["ok"], validation["issues"]
    assert validation["medication_lists_identical"] == 3

    got = json.loads((out / "1.json").read_text(encoding="utf-8"))
    assert got[0]["candidates"] == ["J18.9", "J18"]
    assert got[1]["candidates"] == ["161"], "RxNorm codes must never be dotted"
    assert got[2] == {"text": "sot", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [22, 25]}


def test_derivation_is_deterministic(tmp_path: Path) -> None:
    src = _write_submission(tmp_path / "src", _sample_docs())
    first = tmp_path / "a" / "output"
    second = tmp_path / "b" / "output"
    for out in (first, second):
        derive_submission(
            source_dir=src, output_dir=out, derivation=DERIVE_ICD_DOTTED, expected_documents=3
        )
    assert (first.parent / "output.zip").read_bytes() == (second.parent / "output.zip").read_bytes()


def test_truncate_derivation_is_a_prefix_and_removes_nothing_else(tmp_path: Path) -> None:
    src = _write_submission(tmp_path / "src", _sample_docs())
    out = tmp_path / "t" / "output"
    result = derive_submission(
        source_dir=src,
        output_dir=out,
        derivation=DERIVE_TRUNCATE,
        max_candidates={"CHẨN_ĐOÁN": 1},
        expected_documents=3,
    )
    assert result.candidates_removed == 3
    got = json.loads((out / "1.json").read_text(encoding="utf-8"))
    assert got[0]["candidates"] == ["J189"], "must be a prefix of the source list"
    assert got[1]["candidates"] == ["161"]


# --- governed decoding profiles (§4) -----------------------------------------


def test_the_canonical_default_profile_is_top1() -> None:
    assert DEFAULT_DECODING_PROFILE == "competition_top1"
    assert resolve_decoding_profile(None) is COMPETITION_TOP1
    assert COMPETITION_TOP1.max_candidates == 1


def test_an_unknown_profile_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown decoding profile"):
        resolve_decoding_profile("competition_top9")


def test_top1_emits_at_most_one_medication_candidate() -> None:
    tied = (
        RankedCandidate("A", 10.0, "structured_exact"),
        RankedCandidate("B", 10.0, "structured_exact"),
    )
    assert apply_competition_topk("THUỐC", tied, COMPETITION_TOP1).codes == ("A",)
    # The same input under adaptive keeps both, so the profile is what changed.
    assert apply_competition_topk("THUỐC", tied, COMPETITION_ADAPTIVE).codes == ("A", "B")


def test_top1_emits_at_most_one_diagnosis_candidate() -> None:
    pair = (
        RankedCandidate("A09", 10.0, "supported_specific"),
        RankedCandidate("A090", 8.5, "supported_specific"),
    )
    assert apply_competition_topk("CHẨN_ĐOÁN", pair, COMPETITION_TOP1).codes == ("A09",)
    assert apply_competition_topk("CHẨN_ĐOÁN", pair, COMPETITION_ADAPTIVE).codes == ("A09", "A090")


def test_adaptive_reproduces_the_previous_behaviour() -> None:
    """The default argument must still be the Audit-0058 policy for existing callers."""
    pair = (
        RankedCandidate("A09", 10.0, "supported_specific"),
        RankedCandidate("A090", 8.5, "supported_specific"),
    )
    assert apply_competition_topk("CHẨN_ĐOÁN", pair) == apply_competition_topk(
        "CHẨN_ĐOÁN", pair, COMPETITION_ADAPTIVE
    )


def test_profiles_only_ever_return_an_ordered_prefix() -> None:
    """Requirements 4 and 5: deterministic order, and nothing introduced."""
    offered = (
        RankedCandidate("A", 10.0, "t"),
        RankedCandidate("B", 9.99, "t"),
        RankedCandidate("C", 9.98, "t"),
    )
    for policy in (COMPETITION_TOP1, COMPETITION_ADAPTIVE):
        for _ in range(3):
            codes = apply_competition_topk("CHẨN_ĐOÁN", offered, policy).codes
            assert codes == tuple(c.code for c in offered)[: len(codes)]
            assert set(codes) <= {c.code for c in offered}


def test_switching_profile_changes_only_candidates_not_the_entity() -> None:
    """Requirement 7, at the decoder boundary."""
    from mednorm_vi.confidence_cascade.cascade import CascadeDecision
    from mednorm_vi.linking.models import LinkedCandidate, LinkerResult
    from mednorm_vi.metric_decoder import decode_entities
    from mednorm_vi.resolution.models import (
        STATUS_ACCEPTED,
        BoundaryEvidence,
        EntityHypothesis,
        TypeEvidence,
    )

    hypothesis = EntityHypothesis(
        hypothesis_id="h1",
        document_id="d1",
        start=0,
        end=3,
        text="sot",
        entity_type="CHẨN_ĐOÁN",
        status=STATUS_ACCEPTED,
        chosen_proposal_id="p1",
        source_proposal_ids=("p1",),
        boundary_evidence=BoundaryEvidence("policy", "full", "p1", ()),
        type_evidence=TypeEvidence("CHẨN_ĐOÁN", "E3", ("CHẨN_ĐOÁN",)),
        score=0.9,
    )
    link = LinkerResult(
        "h1",
        (
            LinkedCandidate("A09", 10.0, ("exact",), "snap"),
            LinkedCandidate("A090", 9.99, ("exact",), "snap"),
        ),
    )
    cascade = (CascadeDecision("h1", True, 0.9, ()),)
    top1 = decode_entities((hypothesis,), cascade, (), (link,), topk_policy=COMPETITION_TOP1)
    adaptive = decode_entities(
        (hypothesis,), cascade, (), (link,), topk_policy=COMPETITION_ADAPTIVE
    )
    assert len(top1[0].candidates) == 1
    assert len(adaptive[0].candidates) == 2
    # Everything that is not a candidate must be untouched by the profile.
    for a, b in ((top1[0], adaptive[0]),):
        assert a.hypothesis.text == b.hypothesis.text
        assert a.hypothesis.entity_type == b.hypothesis.entity_type
        assert (a.hypothesis.start, a.hypothesis.end) == (b.hypothesis.start, b.hypothesis.end)
        assert a.assertions == b.assertions
    # And top-1 is a prefix of adaptive: the profile narrows, never reorders.
    assert top1[0].candidates == adaptive[0].candidates[:1]


# --- prepared indices (§10) ---------------------------------------------------


def test_prepared_indexes_load_once_and_respect_config_identity() -> None:
    from mednorm_vi.inference.config import PipelineConfig
    from mednorm_vi.inference.pipeline import PreparedIndexes

    config = PipelineConfig.load(REPO / "configs" / "pipeline" / "full_v1.yaml")
    if not (REPO / config.icd_index).is_file():
        pytest.skip("competition v3 indices not built locally")
    prepared = PreparedIndexes()
    first = prepared.for_config(config)
    assert prepared.loads == 2
    second = prepared.for_config(config)
    assert prepared.loads == 2, "a second call for the same config must not reload"
    assert first[0] is second[0] and first[1] is second[1]

    # A different configured path is a different resource, not a cache hit.
    from dataclasses import replace

    other = replace(config, icd_index="")
    prepared.for_config(other)
    assert prepared.loads == 3
    assert prepared.icd is None
