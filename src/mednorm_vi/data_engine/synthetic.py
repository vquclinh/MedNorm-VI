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


__all__ = ["medication_lab_fixture"]
