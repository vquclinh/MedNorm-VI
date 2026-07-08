"""Submission-layout validation (spec section 16, Appendix A; CONFIRMED layout).

The organizer confirmed the submission layout::

    output.zip
    └── output/
        ├── 1.json
        ├── 2.json
        ├── ...
        └── 100.json

Two entry points:

* ``validate_output_directory`` — an unpacked ``output/`` directory.
* ``validate_submission_zip``   — a packaged ``output.zip``.

Both check: exactly ``1.json``..``100.json`` (no missing IDs, no extra JSON
prediction files), each file UTF-8, each root value a list, and every element
passing organizer-facing entity validation. ZIP validation additionally requires
exactly one top-level ``output/`` directory and rejects unsafe/unexpected nested
paths. Unrelated metadata files are reported (WARNING) but not forbidden unless
they are extra JSON prediction files (which fail).

KB membership of candidate codes remains deferred (no frozen KB yet).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path, PurePosixPath

from ..schemas.constants import N_SUBMISSION_DOCUMENTS
from .organizer import validate_organizer_document
from .results import Severity, ValidationIssue, ValidationResult

SUBMISSION_DIR_NAME = "output"


def expected_filenames(n: int = N_SUBMISSION_DOCUMENTS) -> list[str]:
    """The expected file names: ``1.json`` .. ``{n}.json`` (confirmed form)."""
    return [f"{i}.json" for i in range(1, n + 1)]


def _numeric_key(name: str) -> int:
    return int(name.split(".")[0])


def _validate_json_payload(
    name: str, raw_bytes: bytes
) -> list[ValidationIssue]:
    """Decode UTF-8, parse JSON, require a list root, validate each entity."""
    issues: list[ValidationIssue] = []
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return [
            ValidationIssue(code="submission.not_utf8", message=f"{name!r} is not valid UTF-8",
                            context={"filename": name})
        ]
    try:
        root = json.loads(text)
    except json.JSONDecodeError as exc:
        return [
            ValidationIssue(code="submission.bad_json",
                            message=f"{name!r} is not well-formed JSON: {exc}",
                            context={"filename": name})
        ]
    if not isinstance(root, list):
        issues.append(
            ValidationIssue(code="submission.root_not_list",
                            message=f"{name!r} root value must be a JSON list, got {type(root)!r}",
                            context={"filename": name})
        )
        return issues
    doc_result = validate_organizer_document(root, document_id=name)
    return list(doc_result.issues)


def _compare_fileset(
    present: set[str], n: int
) -> list[ValidationIssue]:
    """Compare present prediction basenames to the expected 1..n set."""
    issues: list[ValidationIssue] = []
    expected = set(expected_filenames(n))
    for missing in sorted(expected - present, key=_numeric_key):
        issues.append(
            ValidationIssue(code="submission.file_missing",
                            message=f"missing expected submission file {missing!r}",
                            context={"filename": missing})
        )
    for extra in sorted(present - expected):
        issues.append(
            ValidationIssue(code="submission.extra_prediction_file",
                            message=f"unexpected extra prediction JSON {extra!r}",
                            context={"filename": extra})
        )
    return issues


def validate_output_directory(
    directory: str | Path,
    *,
    n: int = N_SUBMISSION_DOCUMENTS,
) -> ValidationResult:
    """Validate an unpacked ``output/`` directory against the confirmed layout."""
    issues: list[ValidationIssue] = []
    root = Path(directory)

    if not root.is_dir():
        return ValidationResult.from_issues(
            [ValidationIssue(code="submission.dir_missing",
                             message=f"submission directory {root} does not exist")]
        )

    json_files = {p.name for p in root.iterdir() if p.is_file() and p.suffix == ".json"}
    # Non-JSON metadata files are reported but allowed.
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if p.is_file() and p.suffix != ".json":
            issues.append(
                ValidationIssue(code="submission.metadata_file",
                                message=f"non-prediction metadata file present: {p.name!r}",
                                severity=Severity.WARNING, context={"filename": p.name})
            )
        elif p.is_dir():
            issues.append(
                ValidationIssue(code="submission.unexpected_subdir",
                                message=f"unexpected subdirectory in output/: {p.name!r}",
                                severity=Severity.WARNING, context={"dirname": p.name})
            )

    issues.extend(_compare_fileset(json_files, n))

    for name in sorted(json_files & set(expected_filenames(n)), key=_numeric_key):
        issues.extend(_validate_json_payload(name, (root / name).read_bytes()))

    return ValidationResult.from_issues(issues)


def _is_unsafe_zip_path(name: str) -> bool:
    """True if a zip entry name is unsafe (absolute, drive, backslash, or ..)."""
    if not name or name.startswith("/") or "\\" in name or ":" in name:
        return True
    parts = PurePosixPath(name).parts
    return any(part == ".." for part in parts)


def validate_submission_zip(
    zip_path: str | Path,
    *,
    n: int = N_SUBMISSION_DOCUMENTS,
) -> ValidationResult:
    """Validate a packaged ``output.zip`` against the confirmed layout."""
    issues: list[ValidationIssue] = []
    path = Path(zip_path)
    if not path.is_file():
        return ValidationResult.from_issues(
            [ValidationIssue(code="submission.zip_missing",
                             message=f"submission zip {path} does not exist")]
        )
    if not zipfile.is_zipfile(path):
        return ValidationResult.from_issues(
            [ValidationIssue(code="submission.not_a_zip",
                             message=f"{path} is not a valid ZIP archive")]
        )

    prediction_bytes: dict[str, bytes] = {}
    top_level_dirs: set[str] = set()
    saw_output_entry = False

    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = info.filename
            if _is_unsafe_zip_path(name):
                issues.append(
                    ValidationIssue(code="submission.unsafe_path",
                                    message=f"unsafe/unexpected path in zip: {name!r}",
                                    context={"path": name})
                )
                continue
            # Skip pure directory entries after noting the top-level name.
            parts = PurePosixPath(name).parts
            if not parts:
                continue
            top = parts[0]
            is_dir_entry = name.endswith("/")

            if top == SUBMISSION_DIR_NAME:
                saw_output_entry = True
            elif is_dir_entry or len(parts) > 1:
                # A top-level directory other than output/.
                top_level_dirs.add(top)

            if is_dir_entry:
                continue

            if len(parts) == 1:
                # Top-level file (not under output/).
                if name.endswith(".json"):
                    issues.append(
                        ValidationIssue(code="submission.extra_prediction_file",
                                        message=f"prediction JSON outside output/: {name!r}",
                                        context={"path": name})
                    )
                else:
                    issues.append(
                        ValidationIssue(code="submission.metadata_file",
                                        message=f"top-level metadata file present: {name!r}",
                                        severity=Severity.WARNING, context={"path": name})
                    )
                continue

            if top != SUBMISSION_DIR_NAME:
                # Prediction/data under a wrong top-level directory.
                if name.endswith(".json"):
                    issues.append(
                        ValidationIssue(
                            code="submission.extra_prediction_file",
                            message=f"prediction JSON under wrong top-level dir: {name!r}",
                            context={"path": name},
                        )
                    )
                continue

            # Under output/.
            if len(parts) == 2 and name.endswith(".json"):
                prediction_bytes[parts[1]] = zf.read(name)
            elif len(parts) > 2 and name.endswith(".json"):
                issues.append(
                    ValidationIssue(code="submission.unexpected_nested_path",
                                    message=f"prediction JSON nested too deep: {name!r}",
                                    context={"path": name})
                )
            else:
                # Non-JSON metadata inside output/.
                issues.append(
                    ValidationIssue(code="submission.metadata_file",
                                    message=f"non-prediction file inside output/: {name!r}",
                                    severity=Severity.WARNING, context={"path": name})
                )

    if not saw_output_entry:
        issues.append(
            ValidationIssue(code="submission.missing_output_dir",
                            message="zip has no top-level 'output/' directory")
        )
    for wrong in sorted(top_level_dirs):
        issues.append(
            ValidationIssue(code="submission.wrong_top_level_dir",
                            message=f"unexpected top-level directory in zip: {wrong!r} "
                                    f"(only {SUBMISSION_DIR_NAME!r}/ is allowed)",
                            context={"dirname": wrong})
        )

    issues.extend(_compare_fileset(set(prediction_bytes), n))
    for name in sorted(set(prediction_bytes) & set(expected_filenames(n)), key=_numeric_key):
        issues.extend(_validate_json_payload(name, prediction_bytes[name]))

    return ValidationResult.from_issues(issues)


def write_submission_zip(
    directory: str | Path,
    zip_path: str | Path,
    *,
    n: int = N_SUBMISSION_DOCUMENTS,
) -> Path:
    """Deterministically package ``directory``'s ``1.json``..``{n}.json`` into
    ``output.zip`` with an ``output/`` prefix.

    Entries are written in ascending numeric order for reproducibility. Only the
    expected prediction files are added. Raises ``FileNotFoundError`` if an
    expected file is missing.
    """
    src = Path(directory)
    out = Path(zip_path)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in expected_filenames(n):  # already numeric-ascending
            fpath = src / name
            if not fpath.is_file():
                raise FileNotFoundError(f"expected submission file missing: {fpath}")
            # Fixed date_time for byte-reproducibility of the archive.
            info = zipfile.ZipInfo(filename=f"{SUBMISSION_DIR_NAME}/{name}",
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, fpath.read_bytes())
    return out


__all__ = [
    "expected_filenames",
    "validate_output_directory",
    "validate_submission_zip",
    "write_submission_zip",
    "SUBMISSION_DIR_NAME",
]
