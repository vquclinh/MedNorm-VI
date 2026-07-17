"""Organizer/public-dataset adapters for canonical documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas.constants import TYPE_BY_ORGANIZER_LABEL
from ..schemas.spans import Span
from .models import CanonicalAnnotation, CanonicalDocument


def document_from_text(document_id: str, text: str, *, source_id: str = "") -> CanonicalDocument:
    return CanonicalDocument(document_id=document_id, text=text, source_id=source_id)


def annotations_from_organizer(
    document_id: str, text: str, payload: list[dict[str, Any]], *, source: str
) -> tuple[CanonicalAnnotation, ...]:
    annotations: list[CanonicalAnnotation] = []
    for idx, item in enumerate(payload, 1):
        position = item.get("position", [0, 0])
        start, end = int(position[0]), int(position[1])
        label = str(item.get("type", ""))
        entity_type = TYPE_BY_ORGANIZER_LABEL.get(label, label)
        annotations.append(
            CanonicalAnnotation(
                annotation_id=f"{document_id}-ann-{idx:04d}",
                document_id=document_id,
                span=Span(start, end),
                text=str(item.get("text", text[start:end])),
                entity_type=entity_type,
                assertions=tuple(str(v) for v in item.get("assertions", [])),
                candidates=tuple(str(v) for v in item.get("candidates", [])),
                source=source,
            )
        )
    return tuple(annotations)


def load_manual_gold(text_path: str | Path, annotation_path: str | Path) -> CanonicalDocument:
    text_file = Path(text_path)
    ann_file = Path(annotation_path)
    text = text_file.read_text(encoding="utf-8")
    payload = json.loads(ann_file.read_text(encoding="utf-8"))
    document_id = text_file.stem
    return CanonicalDocument(
        document_id=document_id,
        text=text,
        source_id="manual_gold",
        annotations=annotations_from_organizer(document_id, text, payload, source="manual_gold"),
    )


__all__ = ["annotations_from_organizer", "document_from_text", "load_manual_gold"]
