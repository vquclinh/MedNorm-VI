"""Validation of team annotations and dataset manifests + a small CLI.

Reuses the organizer-facing entity validator (Vietnamese labels, per-type fields)
and adds annotation-specific checks (review status, provenance). The manifest
validator refuses any manifest that claims labeled ``ORGANIZER_TEST`` data.

CLI::

    python -m mednorm_vi.annotation.validation \\
      --dataset data/dev_gold/example \\
      --manifest path/to/manifest.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..evaluation.models import LABELED_PROVENANCE, ORGANIZER_TEST
from ..validator.organizer import validate_organizer_entity
from ..validator.results import Severity, ValidationIssue, ValidationResult
from .models import ReviewStatus

_META_KEYS = ("route_case", "section", "annotator_id", "timestamp", "provenance",
              "review_status", "notes", "source_dataset_id", "guideline_version",
              "document_id")


def validate_annotation_entity(
    entity: dict[str, Any],
    *,
    document_id: str | None = None,
    entity_index: int | None = None,
) -> ValidationResult:
    """Validate one annotation dict: organizer schema + review status + provenance."""
    issues: list[ValidationIssue] = []
    core = {k: v for k, v in entity.items() if k not in _META_KEYS}
    result = validate_organizer_entity(core, document_id=document_id, entity_index=entity_index)
    issues.extend(result.issues)

    status = entity.get("review_status")
    if status is not None and status not in {s.value for s in ReviewStatus}:
        issues.append(ValidationIssue(
            "annotation.bad_review_status",
            f"review_status {status!r} not in {sorted(s.value for s in ReviewStatus)}",
            document_id=document_id, entity_index=entity_index))

    prov = entity.get("provenance")
    if prov is not None:
        if prov == ORGANIZER_TEST:
            issues.append(ValidationIssue(
                "annotation.organizer_test_label",
                "annotation provenance cannot be ORGANIZER_TEST (organizer test has no labels)",
                document_id=document_id, entity_index=entity_index))
        elif prov not in LABELED_PROVENANCE:
            issues.append(ValidationIssue(
                "annotation.bad_provenance",
                f"provenance {prov!r} not in {sorted(LABELED_PROVENANCE)}",
                document_id=document_id, entity_index=entity_index))
    return ValidationResult.from_issues(issues)


def validate_manifest(mapping: dict[str, Any]) -> ValidationResult:
    """Validate a dataset manifest; reject labeled ORGANIZER_TEST claims."""
    issues: list[ValidationIssue] = []
    provenance = mapping.get("provenance")
    labeled = mapping.get("labeled")

    for required in ("dataset_id", "version", "provenance", "labeled"):
        if required not in mapping:
            issues.append(ValidationIssue("manifest.missing_field",
                                          f"manifest missing required field {required!r}"))

    if not isinstance(labeled, bool):
        issues.append(ValidationIssue("manifest.labeled_not_bool",
                                      f"labeled must be a boolean, got {labeled!r}"))

    if provenance == ORGANIZER_TEST:
        if labeled is True:
            issues.append(ValidationIssue(
                "manifest.organizer_test_labeled",
                "manifest declares provenance ORGANIZER_TEST with labeled: true; the "
                "organizer competition test set has NO ground truth and can never be "
                "labeled. Set labeled: false and use it as input only."))
        if mapping.get("includes_organizer_test_inputs") is False:
            issues.append(ValidationIssue(
                "manifest.inconsistent_organizer_flag",
                "provenance ORGANIZER_TEST but includes_organizer_test_inputs is false"))
    elif provenance is not None and provenance not in LABELED_PROVENANCE:
        issues.append(ValidationIssue("manifest.bad_provenance",
                                      f"provenance {provenance!r} not in "
                                      f"{sorted(LABELED_PROVENANCE)} (or ORGANIZER_TEST)"))

    if mapping.get("includes_organizer_test_inputs") is True and labeled is True:
        issues.append(ValidationIssue(
            "manifest.organizer_inputs_labeled",
            "includes_organizer_test_inputs: true requires labeled: false"))
    return ValidationResult.from_issues(issues)


def validate_dataset_dir(directory: str | Path) -> ValidationResult:
    """Validate every annotation file (list of annotation dicts) in a directory."""
    root = Path(directory)
    if not root.is_dir():
        return ValidationResult.from_issues(
            [ValidationIssue("annotation.dir_missing", f"dataset dir {root} not found")])
    result = ValidationResult()
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix == ".json")
    for path in files:
        doc_id = path.stem
        try:
            root_json = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            result = result.merged_with(ValidationResult.from_issues(
                [ValidationIssue("annotation.bad_json", f"{path.name}: {exc}",
                                 document_id=doc_id)]))
            continue
        if not isinstance(root_json, list):
            result = result.merged_with(ValidationResult.from_issues(
                [ValidationIssue("annotation.root_not_list",
                                 f"{path.name} root must be a list", document_id=doc_id)]))
            continue
        for idx, item in enumerate(root_json):
            if not isinstance(item, dict):
                result = result.merged_with(ValidationResult.from_issues(
                    [ValidationIssue("annotation.entity_not_object",
                                     f"{path.name}[{idx}] not an object",
                                     document_id=doc_id, entity_index=idx)]))
                continue
            result = result.merged_with(
                validate_annotation_entity(item, document_id=doc_id, entity_index=idx))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate team annotations / manifest")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args(argv)

    print("MedNorm-VI annotation validation")
    result = validate_dataset_dir(args.dataset)
    if args.manifest:
        import yaml

        mapping = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8")) or {}
        result = result.merged_with(validate_manifest(mapping))

    for issue in result.issues:
        sev = issue.severity.value
        loc = f" (doc={issue.document_id}, entity={issue.entity_index})" \
            if issue.document_id else ""
        stream = sys.stderr if issue.severity is Severity.ERROR else sys.stdout
        print(f"{sev}: {issue.code}: {issue.message}{loc}", file=stream)

    if result.ok:
        print("OK: annotations valid")
        return 0
    print(f"FAILED: {len(result.errors)} error(s)", file=sys.stderr)
    return 1


__all__ = [
    "validate_annotation_entity",
    "validate_manifest",
    "validate_dataset_dir",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
