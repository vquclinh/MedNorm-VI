"""Inter-annotator agreement + adjudication tests."""

from __future__ import annotations

from mednorm_vi.annotation.adjudication import make_adjudication, validate_adjudication
from mednorm_vi.annotation.agreement import corpus_agreement, document_agreement
from mednorm_vi.annotation.models import AnnotationEntity, ReviewStatus


def _a(text, etype, s, e, asserts=(), cands=()):
    return AnnotationEntity("1", text, etype, s, e, assertions=asserts, candidates=cands)


def test_full_agreement() -> None:
    a = [_a("sốt", "TRIỆU_CHỨNG", 0, 3), _a("ho", "TRIỆU_CHỨNG", 5, 7)]
    b = [_a("sốt", "TRIỆU_CHỨNG", 0, 3), _a("ho", "TRIỆU_CHỨNG", 5, 7)]
    s = document_agreement("1", a, b)
    assert s.span_agreement == 1.0
    assert s.type_agreement == 1.0


def test_span_disagreement() -> None:
    a = [_a("sốt", "TRIỆU_CHỨNG", 0, 3)]
    b = [_a("sốt", "TRIỆU_CHỨNG", 1, 4)]  # different span
    s = document_agreement("1", a, b)
    assert s.n_span_agree == 0
    assert s.span_agreement == 0.0


def test_type_disagreement_on_same_span() -> None:
    a = [_a("Ho khan", "TRIỆU_CHỨNG", 0, 7)]
    b = [_a("Ho khan", "CHẨN_ĐOÁN", 0, 7)]
    s = document_agreement("1", a, b)
    assert s.n_span_agree == 1
    assert s.n_type_agree == 0


def test_assertion_and_candidate_agreement() -> None:
    a = [_a("dx", "CHẨN_ĐOÁN", 0, 2, asserts=("isHistorical",), cands=("A",))]
    b = [_a("dx", "CHẨN_ĐOÁN", 0, 2, asserts=(), cands=("A",))]
    s = document_agreement("1", a, b)
    assert s.n_candidate_agree == 1
    assert s.n_assertion_agree == 0


def test_corpus_agreement_aggregates() -> None:
    s1 = document_agreement("1", [_a("x", "THUỐC", 0, 1, cands=("1",))],
                            [_a("x", "THUỐC", 0, 1, cands=("1",))])
    agg = corpus_agreement([s1])
    assert agg["span_agreement"] == 1.0
    assert agg["n_documents"] == 1.0


def test_adjudication_forces_adjudicated_status() -> None:
    resolved = _a("sốt", "TRIỆU_CHỨNG", 0, 3)
    rec = make_adjudication(document_id="1", annotator_a="A", annotator_b="B",
                            disagreement_reason="type", adjudicated_result=resolved,
                            adjudicator_id="C", guideline_version="v1")
    assert rec.adjudicated_result.review_status is ReviewStatus.ADJUDICATED
    assert validate_adjudication(rec).ok


def test_adjudication_same_annotator_rejected() -> None:
    resolved = _a("sốt", "TRIỆU_CHỨNG", 0, 3)
    rec = make_adjudication(document_id="1", annotator_a="A", annotator_b="A",
                            disagreement_reason="x", adjudicated_result=resolved,
                            adjudicator_id="C", guideline_version="v1")
    r = validate_adjudication(rec)
    assert not r.ok
    assert any(i.code == "adjudication.same_annotator" for i in r.errors)
