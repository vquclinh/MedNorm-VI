"""Controlled synthetic document generation for offline tests and training dry runs."""

from __future__ import annotations

from ..schemas.spans import Span
from .models import CanonicalAnnotation, CanonicalDocument


def medication_lab_fixture(document_id: str = "synthetic-1") -> CanonicalDocument:
    text = "Thuốc: paracetamol 500mg\nXét nghiệm: glucose 7 mmol/L\n"
    med_start = text.index("paracetamol")
    lab_start = text.index("glucose")
    value_start = text.index("7 mmol/L")
    return CanonicalDocument(
        document_id=document_id,
        text=text,
        source_id="synthetic",
        template_id="med_lab",
        annotations=(
            CanonicalAnnotation(
                annotation_id=f"{document_id}-a1",
                document_id=document_id,
                span=Span(med_start, med_start + len("paracetamol 500mg")),
                text="paracetamol 500mg",
                entity_type="MEDICATION",
                source="synthetic",
            ),
            CanonicalAnnotation(
                annotation_id=f"{document_id}-a2",
                document_id=document_id,
                span=Span(lab_start, lab_start + len("glucose")),
                text="glucose",
                entity_type="TEST_NAME",
                source="synthetic",
            ),
            CanonicalAnnotation(
                annotation_id=f"{document_id}-a3",
                document_id=document_id,
                span=Span(value_start, value_start + len("7 mmol/L")),
                text="7 mmol/L",
                entity_type="TEST_RESULT",
                source="synthetic",
            ),
        ),
    )


def _ann(doc_id: str, i: int, text: str, start: int, etype: str,
         assertions: tuple[str, ...] = ()) -> CanonicalAnnotation:
    return CanonicalAnnotation(
        annotation_id=f"{doc_id}-a{i}", document_id=doc_id,
        span=Span(start, start + len(text)), text=text, entity_type=etype,
        assertions=assertions, source="synthetic")


def _doc(doc_id: str, template: str, text: str,
         spans: list[tuple[str, str, tuple[str, ...]]]) -> CanonicalDocument:
    anns = []
    cursor = 0
    for i, (surface, etype, asserts) in enumerate(spans, 1):
        start = text.index(surface, cursor)
        anns.append(_ann(doc_id, i, surface, start, etype, asserts))
        cursor = start + len(surface)
    doc = CanonicalDocument(
        document_id=doc_id, text=text, source_id="synthetic", template_id=template,
        metadata={"provenance": "synthetic", "template_id": template, "seed": "0"},
        annotations=tuple(anns))
    for a in doc.annotations:                       # offset invariant
        assert text[a.span.start:a.span.end] == a.text
    return doc


def synthetic_fixture_examples() -> tuple[CanonicalDocument, ...]:
    """Tiny deterministic synthetic fixtures for the HIGH-PRIORITY missing
    supervision (TEST_NAME/TEST_RESULT, medication boundaries, assertions).

    These are for tests and training dry-runs only — the large synthetic corpus is
    a Colab/Drive-backed step. Synthetic data must never enter validation or
    internal_test; the governed-corpus builder does not read these.
    """
    return (
        # TEST_NAME + TEST_RESULT value-only
        _doc("syn-lab-valueonly", "lab_value_only", "WBC: 14,43\n",
             [("WBC", "TEST_NAME", ()), ("14,43", "TEST_RESULT", ())]),
        # TEST_RESULT value + unit
        _doc("syn-lab-valueunit", "lab_value_unit", "Glucose: 7.8 mmol/L\n",
             [("Glucose", "TEST_NAME", ()), ("7.8 mmol/L", "TEST_RESULT", ())]),
        # comparison result
        _doc("syn-lab-comparison", "lab_comparison", "CRP: < 5 mg/L\n",
             [("CRP", "TEST_NAME", ()), ("< 5 mg/L", "TEST_RESULT", ())]),
        # qualitative result
        _doc("syn-lab-qualitative", "lab_qualitative", "HIV: âm tính\n",
             [("HIV", "TEST_NAME", ()), ("âm tính", "TEST_RESULT", ())]),
        # medication list boundary (ingredient + strength + route + frequency)
        _doc("syn-med-list", "med_list_full", "1. amlodipine 10 mg po daily\n",
             [("amlodipine 10 mg po daily", "MEDICATION", ())]),
        # assertions: negation, family, historical
        _doc("syn-assert-neg", "assert_negation", "Không sốt, không ho.\n",
             [("sốt", "SYMPTOM", ("isNegated",)), ("ho", "SYMPTOM", ("isNegated",))]),
        _doc("syn-assert-family", "assert_family", "Mẹ bị đái tháo đường.\n",
             [("đái tháo đường", "DIAGNOSIS", ("isFamily",))]),
        _doc("syn-assert-hist", "assert_historical", "Tiền sử tăng huyết áp.\n",
             [("tăng huyết áp", "DIAGNOSIS", ("isHistorical",))]),
    )


__all__ = ["medication_lab_fixture", "synthetic_fixture_examples"]
