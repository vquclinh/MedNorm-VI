"""Loading + validation of ground-truth and prediction directories.

* Predictions are validated STRICTLY as organizer submission entities.
* Ground truth is team-owned labeled data; each entity may additionally carry
  optional ``route_case`` / ``section`` metadata (stripped before organizer
  validation and used only for diagnostic grouping).
* Ground-truth provenance is resolved from an optional ``manifest.yaml`` or an
  explicit override. ``ORGANIZER_TEST`` can never be a labeled provenance — the
  organizer competition test set has no ground truth.

Document id = file stem. Files are read in deterministic (numeric-aware) order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..validator.organizer import validate_organizer_entity
from ..validator.results import Severity, ValidationIssue, ValidationResult
from .models import (
    ORGANIZER_TEST,
    EvaluationDocument,
    EvaluationEntity,
    Provenance,
    parse_provenance,
)

_META_KEYS = ("route_case", "section")


def _numeric_aware(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), stem) if stem.isdigit() else (1 << 60, stem)


def _json_files(directory: Path) -> list[Path]:
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix == ".json"),
                  key=_numeric_aware)


def _entity_from_dict(item: dict[str, Any]) -> EvaluationEntity:
    pos = item.get("position", [0, 0])
    start, end = (int(pos[0]), int(pos[1])) if isinstance(pos, (list, tuple)) and len(pos) == 2 \
        else (0, 0)
    return EvaluationEntity(
        text=str(item.get("text", "")),
        type=str(item.get("type", "")),
        start=start,
        end=end,
        assertions=tuple(item.get("assertions", []) or []),
        candidates=tuple(item.get("candidates", []) or []),
        route_case=item.get("route_case"),
        section=item.get("section"),
    )


def _load_document(
    path: Path, *, provenance: Provenance | None, allow_metadata: bool
) -> tuple[EvaluationDocument | None, ValidationResult]:
    doc_id = path.stem
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None, ValidationResult.from_issues(
            [ValidationIssue("eval.not_utf8", f"{path.name} is not UTF-8", document_id=doc_id)]
        )
    try:
        root = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, ValidationResult.from_issues(
            [ValidationIssue("eval.bad_json", f"{path.name}: {exc}", document_id=doc_id)]
        )
    if not isinstance(root, list):
        return None, ValidationResult.from_issues(
            [ValidationIssue("eval.root_not_list",
                             f"{path.name} root must be a JSON list", document_id=doc_id)]
        )

    issues: list[ValidationIssue] = []
    entities: list[EvaluationEntity] = []
    for idx, item in enumerate(root):
        if not isinstance(item, dict):
            issues.append(ValidationIssue("eval.entity_not_object",
                                          f"{path.name}[{idx}] is not an object",
                                          document_id=doc_id, entity_index=idx))
            continue
        core = {k: v for k, v in item.items() if not (allow_metadata and k in _META_KEYS)}
        result = validate_organizer_entity(core, document_id=doc_id, entity_index=idx)
        issues.extend(result.issues)
        if result.ok:
            entities.append(_entity_from_dict(item))
    doc = EvaluationDocument(document_id=doc_id, entities=tuple(entities),
                             provenance=provenance)
    return doc, ValidationResult.from_issues(issues)


def load_predictions(
    directory: str | Path,
) -> tuple[dict[str, EvaluationDocument], ValidationResult]:
    """Load + strictly validate prediction documents (organizer submission shape)."""
    root = Path(directory)
    if not root.is_dir():
        return {}, ValidationResult.from_issues(
            [ValidationIssue("eval.pred_dir_missing", f"predictions directory {root} not found")]
        )
    docs: dict[str, EvaluationDocument] = {}
    result = ValidationResult()
    for path in _json_files(root):
        doc, res = _load_document(path, provenance=None, allow_metadata=False)
        result = result.merged_with(res)
        if doc is not None:
            docs[doc.document_id] = doc
    return docs, result


def resolve_ground_truth_provenance(
    directory: Path, *, override: Provenance | None
) -> tuple[Provenance | None, ValidationResult]:
    """Resolve provenance from override or an optional manifest.yaml. Reject ORGANIZER_TEST."""
    if override is not None:
        return override, ValidationResult()
    manifest = directory / "manifest.yaml"
    if not manifest.is_file():
        return None, ValidationResult.from_issues(
            [ValidationIssue(
                "eval.provenance_unknown",
                f"ground-truth provenance unknown: provide --provenance or a "
                f"manifest.yaml in {directory}",
            )]
        )
    import yaml

    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    prov_str = str(data.get("provenance", ""))
    labeled = bool(data.get("labeled", True))
    if prov_str == ORGANIZER_TEST:
        return None, ValidationResult.from_issues(
            [ValidationIssue(
                "eval.organizer_test_labeled",
                "manifest declares ORGANIZER_TEST provenance; organizer test inputs "
                "have no ground truth and cannot be used as labeled evaluation data.",
            )]
        )
    if not labeled:
        return None, ValidationResult.from_issues(
            [ValidationIssue("eval.manifest_unlabeled",
                             "manifest declares labeled: false; cannot evaluate as ground truth.")]
        )
    try:
        return parse_provenance(prov_str), ValidationResult()
    except ValueError as exc:
        return None, ValidationResult.from_issues(
            [ValidationIssue("eval.bad_provenance", str(exc))]
        )


def load_ground_truth(
    directory: str | Path, *, provenance: Provenance | None = None
) -> tuple[dict[str, EvaluationDocument], Provenance | None, ValidationResult]:
    """Load + validate team-owned labeled ground-truth documents."""
    root = Path(directory)
    if not root.is_dir():
        return {}, None, ValidationResult.from_issues(
            [ValidationIssue("eval.gt_dir_missing", f"ground-truth directory {root} not found")]
        )
    prov, prov_result = resolve_ground_truth_provenance(root, override=provenance)
    if not prov_result.ok:
        return {}, prov, prov_result
    docs: dict[str, EvaluationDocument] = {}
    result = prov_result
    for path in _json_files(root):
        if path.name == "manifest.yaml":
            continue
        doc, res = _load_document(path, provenance=prov, allow_metadata=True)
        result = result.merged_with(res)
        if doc is not None:
            docs[doc.document_id] = doc
    if not docs and result.ok:
        result = result.merged_with(ValidationResult.from_issues(
            [ValidationIssue("eval.no_documents", f"no .json documents found in {root}",
                             severity=Severity.WARNING)]
        ))
    return docs, prov, result


__all__ = [
    "load_predictions",
    "load_ground_truth",
    "resolve_ground_truth_provenance",
]
