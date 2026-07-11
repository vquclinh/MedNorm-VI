"""Export accepted-gold annotations to evaluator ground-truth files.

Only ADJUDICATED (or explicitly approved GOLD) annotations are exported by
default. Output is the organizer entity shape (Vietnamese label, per-type fields)
plus optional ``route_case``/``section`` metadata that the evaluator uses for
diagnostic grouping. A ``manifest.yaml`` recording provenance is written too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas.constants import ORGANIZER_FIELDS_BY_TYPE, TYPE_BY_ORGANIZER_LABEL
from .models import AnnotationEntity


def annotation_to_gt_dict(annotation: AnnotationEntity) -> dict[str, Any]:
    """Organizer-shaped dict for one annotation, plus optional route/section metadata."""
    internal = TYPE_BY_ORGANIZER_LABEL.get(annotation.type)
    if internal is None:
        raise ValueError(f"unknown organizer label {annotation.type!r}")
    available: dict[str, Any] = {
        "text": annotation.text,
        "type": annotation.type,
        "assertions": list(annotation.assertions),
        "candidates": list(annotation.candidates),
        "position": [annotation.start, annotation.end],
    }
    out = {name: available[name] for name in ORGANIZER_FIELDS_BY_TYPE[internal]}
    if annotation.route_case is not None:
        out["route_case"] = annotation.route_case
    if annotation.section is not None:
        out["section"] = annotation.section
    return out


def group_accepted_gold(
    annotations: list[AnnotationEntity], *, only_accepted: bool = True
) -> dict[str, list[AnnotationEntity]]:
    """Group annotations by document id, filtering to accepted gold by default."""
    grouped: dict[str, list[AnnotationEntity]] = {}
    for a in annotations:
        if only_accepted and not a.is_accepted_gold():
            continue
        grouped.setdefault(a.document_id, []).append(a)
    return grouped


def export_gold_dataset(
    annotations: list[AnnotationEntity],
    out_dir: str | Path,
    *,
    provenance: str = "GOLD",
    dataset_id: str = "dev_gold_export",
    only_accepted: bool = True,
) -> list[str]:
    """Write per-document GT files + manifest.yaml. Returns written filenames."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    grouped = group_accepted_gold(annotations, only_accepted=only_accepted)

    written: list[str] = []
    n_entities = 0
    for doc_id in sorted(grouped, key=lambda s: (int(s), s) if s.isdigit() else (1 << 60, s)):
        rows = [annotation_to_gt_dict(a) for a in grouped[doc_id]]
        n_entities += len(rows)
        text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        (root / f"{doc_id}.json").write_text(text, encoding="utf-8")
        written.append(f"{doc_id}.json")

    manifest = (
        f"dataset_id: {dataset_id}\n"
        "version: 1\n"
        f"provenance: {provenance}\n"
        "labeled: true\n"
        f"n_documents: {len(grouped)}\n"
        f"n_entities: {n_entities}\n"
        "includes_organizer_test_inputs: false\n"
    )
    (root / "manifest.yaml").write_text(manifest, encoding="utf-8")
    return written


__all__ = ["annotation_to_gt_dict", "group_accepted_gold", "export_gold_dataset"]
