"""Derive a submission variant from an existing one, without re-running inference.

Two milestones needed this and both did it with an untracked ad-hoc script, which
Audit 0059 recorded as a `FINAL_SUBMISSION_BLOCKER`: profiles A and B were produced by
truncating Profile C's JSON, and those ZIPs could not be regenerated from the
repository. This module replaces that.

It supports exactly two derivations, because these are the two that are provably
safe to perform on an already-validated submission:

``icd_dotted``
    rewrite diagnosis candidate codes into dotted form (Audit 0060's probe for the
    unresolved ``UNRES-ICD-DOTTED`` policy). Concept identity is unchanged; only the
    character form moves.

``truncate``
    keep at most *n* candidates per type. Valid only because
    ``apply_competition_topk`` returns an order-preserving sub-sequence, so a stricter
    policy's output is a **prefix** of a looser one's — the derived file is exactly
    what the stricter policy would have emitted.

Everything else is copied byte-for-byte. `text`, `type`, `position` and `assertions`
are never touched, no entity is added or removed, and entity order is preserved. A
derivation that cannot honour that refuses rather than degrades.

Re-running inference to obtain these variants would be worse than wasteful — it would
make the comparison unsound, because two runs could differ for reasons other than the
one variable under test.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas.constants import ORGANIZER_LABEL_BY_TYPE
from .code_format import CODE_FORMAT_VERSION, TransformReport, transform_diagnosis_codes
from .packaging import package_output_zip

DERIVATION_VERSION = "submission-derivation-v1"

DIAGNOSIS_LABEL = ORGANIZER_LABEL_BY_TYPE["DIAGNOSIS"]
MEDICATION_LABEL = ORGANIZER_LABEL_BY_TYPE["MEDICATION"]

DERIVE_ICD_DOTTED = "icd_dotted"
DERIVE_TRUNCATE = "truncate"


class DerivationRefused(RuntimeError):
    """The derivation could not be performed without changing something it must not."""


@dataclass(frozen=True, slots=True)
class DerivationResult:
    """Aggregate outcome. Carries counts and codes, never clinical text."""

    derivation: str
    source_dir: str
    output_dir: str
    documents: int
    entities: int
    candidate_lists_changed: int = 0
    candidates_removed: int = 0
    code_report: TransformReport | None = None
    malformed: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "derivation_version": DERIVATION_VERSION,
            "derivation": self.derivation,
            "source_dir": self.source_dir,
            "output_dir": self.output_dir,
            "documents": self.documents,
            "entities": self.entities,
            "candidate_lists_changed": self.candidate_lists_changed,
            "candidates_removed": self.candidates_removed,
            "malformed_codes": list(self.malformed),
        }
        if self.code_report is not None:
            payload["code_format"] = self.code_report.as_dict()
        return payload


def _read_documents(
    source_dir: Path, *, styles: dict[int, str] | None = None
) -> dict[int, list[dict[str, Any]]]:
    docs: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(source_dir.glob("*.json"), key=lambda p: int(p.stem)):
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise DerivationRefused(f"{path.name}: document root is not a JSON list")
        docs[int(path.stem)] = payload
        if styles is not None:
            styles[int(path.stem)] = _detect_style(raw)
    if not docs:
        raise DerivationRefused(f"no *.json documents under {source_dir}")
    return docs


def _detect_style(sample: str) -> str:
    """Whether a source document was written compact or with the canonical indent.

    The canonical serializer writes ``indent=2``; the ad-hoc script that produced the
    Audit-0059 baseline profiles wrote compact JSON, and the leaderboard accepted both,
    so whitespace is not graded. A derivation must nonetheless *preserve* whichever
    style its source used: re-flowing the JSON would make every byte differ and destroy
    the claim that the dot is the only change.
    """
    return "indent" if sample.lstrip().startswith("[\n") else "compact"


def _serialize(entities: Sequence[Mapping[str, Any]], *, style: str) -> str:
    """Serialize deterministically in the source document's own style."""
    if style == "indent":
        return json.dumps(list(entities), ensure_ascii=False, indent=2) + "\n"
    return json.dumps(list(entities), ensure_ascii=False, separators=(",", ":")) + "\n"


def derive_submission(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    derivation: str,
    max_candidates: Mapping[str, int] | None = None,
    expected_documents: int = 100,
) -> DerivationResult:
    """Write a derived ``output/`` directory and its ZIP beside it."""
    src = Path(source_dir)
    out = Path(output_dir)
    styles: dict[int, str] = {}
    docs = _read_documents(src, styles=styles)

    entities = 0
    lists_changed = 0
    removed = 0
    report = TransformReport()
    malformed: list[str] = []
    written: dict[str, str] = {}

    for doc_id, ents in sorted(docs.items()):
        new_ents: list[dict[str, Any]] = []
        for entity in ents:
            entities += 1
            item = dict(entity)
            if "candidates" in item:
                original = list(item["candidates"])
                codes = original
                if derivation == DERIVE_ICD_DOTTED:
                    if item.get("type") == DIAGNOSIS_LABEL:
                        codes, sub = transform_diagnosis_codes(codes)
                        report = TransformReport(
                            report.transformed + sub.transformed,
                            report.unchanged_category + sub.unchanged_category,
                            report.unchanged_already_dotted + sub.unchanged_already_dotted,
                            report.malformed + sub.malformed,
                        )
                        malformed.extend(sub.malformed)
                elif derivation == DERIVE_TRUNCATE:
                    limit = (max_candidates or {}).get(str(item.get("type")))
                    if limit is not None and len(codes) > limit:
                        codes = codes[:limit]
                        removed += len(original) - len(codes)
                else:
                    raise DerivationRefused(f"unknown derivation: {derivation!r}")
                if codes != original:
                    lists_changed += 1
                item["candidates"] = codes
            new_ents.append(item)
        if len(new_ents) != len(ents):
            raise DerivationRefused(f"document {doc_id}: entity count changed")
        written[f"{doc_id}.json"] = _serialize(new_ents, style=styles.get(doc_id, "indent"))

    if malformed:
        raise DerivationRefused(
            f"refusing to derive: {len(malformed)} malformed ICD code(s) encountered, "
            f"first few {sorted(set(malformed))[:5]}"
        )

    out.mkdir(parents=True, exist_ok=True)
    for name, text in written.items():
        (out / name).write_text(text, encoding="utf-8")
    package_output_zip(out, out.parent / "output.zip", expected_count=expected_documents)

    return DerivationResult(
        derivation=derivation,
        source_dir=src.as_posix(),
        output_dir=out.as_posix(),
        documents=len(docs),
        entities=entities,
        candidate_lists_changed=lists_changed,
        candidates_removed=removed,
        code_report=report if derivation == DERIVE_ICD_DOTTED else None,
        malformed=tuple(sorted(set(malformed))),
    )


def validate_derivation(
    *, source_dir: str | Path, derived_dir: str | Path, derivation: str
) -> dict[str, Any]:
    """Prove the derived output differs from its source only in the intended way."""
    src_docs = _read_documents(Path(source_dir))
    new_docs = _read_documents(Path(derived_dir))
    issues: list[str] = []
    if set(src_docs) != set(new_docs):
        issues.append("document id sets differ")
    checked = med_identical = 0
    for doc_id in sorted(set(src_docs) & set(new_docs)):
        a, b = src_docs[doc_id], new_docs[doc_id]
        if len(a) != len(b):
            issues.append(f"{doc_id}: entity count differs")
            continue
        for x, y in zip(a, b, strict=True):
            checked += 1
            for key in ("text", "type", "position", "assertions"):
                if x.get(key) != y.get(key):
                    issues.append(f"{doc_id}: {key} changed")
            if x.get("type") == MEDICATION_LABEL and x.get("candidates") != y.get("candidates"):
                issues.append(f"{doc_id}: medication candidates changed")
            elif x.get("type") == MEDICATION_LABEL:
                med_identical += 1
            if "candidates" in x:
                fmt_only = derivation == DERIVE_ICD_DOTTED
                if fmt_only and len(x["candidates"]) != len(y["candidates"]):
                    issues.append(f"{doc_id}: candidate count changed under a format-only run")
                if derivation == DERIVE_ICD_DOTTED and x.get("type") == DIAGNOSIS_LABEL:
                    if [c.replace(".", "") for c in y["candidates"]] != [
                        c.replace(".", "") for c in x["candidates"]
                    ]:
                        issues.append(f"{doc_id}: diagnosis identity changed, not just the dot")
    return {
        "derivation_version": DERIVATION_VERSION,
        "code_format_version": CODE_FORMAT_VERSION,
        "derivation": derivation,
        "documents": len(new_docs),
        "entities_checked": checked,
        "medication_lists_identical": med_identical,
        "ok": not issues,
        "issues": issues[:20],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--derivation", required=True, choices=(DERIVE_ICD_DOTTED, DERIVE_TRUNCATE))
    parser.add_argument("--max-diagnosis-candidates", type=int, default=None)
    parser.add_argument("--max-medication-candidates", type=int, default=None)
    parser.add_argument("--expected-documents", type=int, default=100)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--validation-report", default=None)
    args = parser.parse_args(argv)

    limits: dict[str, int] = {}
    if args.max_diagnosis_candidates is not None:
        limits[DIAGNOSIS_LABEL] = args.max_diagnosis_candidates
    if args.max_medication_candidates is not None:
        limits[MEDICATION_LABEL] = args.max_medication_candidates

    result = derive_submission(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        derivation=args.derivation,
        max_candidates=limits or None,
        expected_documents=args.expected_documents,
    )
    validation = validate_derivation(
        source_dir=args.source_dir, derived_dir=args.output_dir, derivation=args.derivation
    )
    if args.manifest:
        Path(args.manifest).write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.validation_report:
        Path(args.validation_report).write_text(
            json.dumps(validation, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    json.dump(
        {"result": result.as_dict(), "validation": validation},
        sys.stdout,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DERIVATION_VERSION",
    "DERIVE_ICD_DOTTED",
    "DERIVE_TRUNCATE",
    "DerivationRefused",
    "DerivationResult",
    "derive_submission",
    "main",
    "validate_derivation",
]
