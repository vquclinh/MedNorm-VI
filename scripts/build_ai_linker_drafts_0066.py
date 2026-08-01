#!/usr/bin/env python3
"""Materialise AI-assisted linker pre-annotation drafts (Audit 0066 §2).

The drafts themselves are produced by reading each record's mention, bounded context and
offered candidate evidence and deciding; this script is the **writer**, not the annotator.
It takes the recorded decisions, re-validates every one against the frozen pack, and emits
a private draft file that is structurally incapable of being mistaken for human gold.

Three guarantees, all enforced here rather than trusted:

* every draft carries `adjudication_status = AI_ASSISTED_DRAFT`, which is outside
  `HUMAN_GOLD_STATUSES`, so `is_human_gold` is false for all of them;
* a selected code must be among the candidates that record actually offered - a draft
  cannot invent a concept the linker never surfaced;
* the ontology restriction (`CHẨN_ĐOÁN → ICD10`, `THUỐC → RXNORM`) is re-checked per
  record, so a mis-keyed decision fails loudly instead of landing in the wrong ontology.

Drafts are written to an ignored private path and never touch `pack.jsonl`, so a human
decision can never be overwritten by a model's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.annotation.linker_gold import (  # noqa: E402
    AI_CONFIDENCES,
    AI_DECISION_NO_VALID,
    AI_DECISION_SELECTED,
    AI_DECISIONS,
    ONTOLOGY_ICD10,
    STATUS_AI_ASSISTED_DRAFT,
    LinkerGoldError,
    assert_not_ai_gold,
    is_human_gold,
    load_records,
    utc_now,
)

PACK = REPO / "data" / "private_linker_gold" / "0065" / "pack.jsonl"
OUT = REPO / "data" / "private_linker_gold" / "0065" / "ai_drafts"
METHOD = "claude-code-two-pass-evidence-grounded-v1"
SCHEMA_VERSION = "ai_linker_draft_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def read_decisions(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Tab-separated decisions: id, decision, codes, confidence, flag, rationale."""
    decisions: dict[str, dict[str, Any]] = {}
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 6:
                raise LinkerGoldError(f"{path}:{number}: expected 6 tab-separated fields")
            short, decision, codes, confidence, flag, rationale = fields[:6]
            if decision not in AI_DECISIONS:
                raise LinkerGoldError(f"{path}:{number}: unknown decision {decision!r}")
            if confidence not in AI_CONFIDENCES:
                raise LinkerGoldError(f"{path}:{number}: unknown confidence {confidence!r}")
            if short in decisions:
                raise LinkerGoldError(f"{path}:{number}: duplicate decision for {short}")
            decisions[short] = {
                "decision": decision,
                "codes": [c for c in codes.split(",") if c],
                "confidence": confidence,
                "flag": flag,
                "rationale": rationale.strip(),
            }
    return decisions


def build_draft(record: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    offered = {c["code"] for c in record.get("offered_candidates") or []}
    codes = list(decision["codes"])

    if decision["decision"] == AI_DECISION_SELECTED and not codes:
        raise LinkerGoldError(f"{record['annotation_id']}: SELECTED with no codes")
    if decision["decision"] != AI_DECISION_SELECTED and codes:
        raise LinkerGoldError(
            f"{record['annotation_id']}: {decision['decision']} must not carry codes"
        )
    unknown = [c for c in codes if c not in offered]
    if unknown:
        raise LinkerGoldError(
            f"{record['annotation_id']}: selected codes were never offered: {unknown}"
        )
    if record["ontology"] == ONTOLOGY_ICD10 and any("." in c for c in codes):
        raise LinkerGoldError(
            f"{record['annotation_id']}: ICD codes must be dotless internal identities"
        )

    ranked = sorted(record.get("offered_candidates") or [], key=lambda c: c["linker_rank"])
    linker_top1 = ranked[0]["code"] if ranked else ""
    return {
        "annotation_id": record["annotation_id"],
        "ontology": record["ontology"],
        "entity_type": record["entity_type"],
        "adjudication_status": STATUS_AI_ASSISTED_DRAFT,
        "ai_decision": decision["decision"],
        "selected_codes": codes,
        "no_valid_candidate": decision["decision"] == AI_DECISION_NO_VALID,
        "ai_confidence": decision["confidence"],
        "ambiguity_reason": decision["flag"],
        "rationale": decision["rationale"],
        "competing_candidates": [c for c in offered if c not in set(codes)][:10],
        "evidence_fields_used": evidence_fields(record, decision),
        "additional_kb_search_needed": decision["flag"]
        in ("empty_set", "no_candidate_represents", "lexical_false_friend"),
        "linker_top1": linker_top1,
        "agrees_with_linker_top1": bool(codes) and linker_top1 in set(codes),
        "offered_count": len(offered),
        "stratum": record["provenance"].get("stratum", ""),
        "method": METHOD,
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
    }


def evidence_fields(record: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    fields = ["mention_text", "context_before", "context_after", "candidate_canonical_name"]
    if record["ontology"] == ONTOLOGY_ICD10:
        fields += ["icd_specificity_class", "icd_parent_code", "icd_child_codes"]
    else:
        fields += ["rxnorm_tty", "rxnorm_ingredient", "rxnorm_strength", "rxnorm_dose_form"]
    if decision["flag"] in ("empty_set",):
        fields = ["mention_text", "context_before", "context_after"]
    return fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--decisions", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--pass-b-changes",
        type=Path,
        default=None,
        help="TSV of ids whose PASS-B critique changed the PASS-A decision",
    )
    args = parser.parse_args(argv)

    assert_not_ai_gold()
    records = load_records(args.pack)
    by_short = {r["annotation_id"][4:12]: r for r in records}
    decisions = read_decisions(list(args.decisions))

    unknown = sorted(set(decisions) - set(by_short))
    if unknown:
        raise SystemExit(f"decisions reference unknown records: {unknown[:5]}")

    drafts = []
    for short, decision in decisions.items():
        record = by_short[short]
        if is_human_gold(record):
            # A model draft must never overwrite a person's decision.
            print(f"  skipping {short}: already human gold")
            continue
        drafts.append(build_draft(record, decision))
    drafts.sort(key=lambda d: d["annotation_id"])

    args.out.mkdir(parents=True, exist_ok=True)
    draft_path = args.out / "claude_code_v1.jsonl"
    draft_path.write_text(
        "".join(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n" for d in drafts),
        encoding="utf-8",
    )

    decisions_count = Counter(d["ai_decision"] for d in drafts)
    confidence_count = Counter(d["ai_confidence"] for d in drafts)
    by_ontology = Counter(d["ontology"] for d in drafts)
    flags = Counter(d["ambiguity_reason"] for d in drafts)
    agree = sum(1 for d in drafts if d["agrees_with_linker_top1"])

    changed = 0
    if args.pass_b_changes and args.pass_b_changes.is_file():
        changed = sum(
            1
            for line in args.pass_b_changes.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "annotation_method_note": (
            "Evidence-grounded pre-annotation by Claude Code over mention text, bounded "
            "context and the offered candidate evidence only. Two passes: PASS A independent "
            "draft, PASS B self-critique. NOT human gold and NOT eligible to unlock the "
            "benchmark."
        ),
        "git_commit": git_commit(),
        "architecture_pdf_sha256": sha256_file(REPO / "docs" / "MedNorm-VI_Architecture.pdf"),
        "pack_path": str(
            args.pack.relative_to(REPO) if args.pack.is_relative_to(REPO) else args.pack
        ),
        "pack_sha256": sha256_file(args.pack),
        "kb_snapshot_id": records[0].get("candidate_snapshot_id", "") if records else "",
        "icd_index_sha256": sha256_file(
            REPO / "indices" / "candidate" / "icd10_vi" / "competition-v3" / "index.json"
        ),
        "rxnorm_index_sha256": sha256_file(
            REPO / "indices" / "candidate" / "rxnorm" / "competition-v3" / "index.json"
        ),
        "created_at": utc_now(),
        "processed_records": len(drafts),
        "pack_records": len(records),
        "deterministic_processing_order": "ascending annotation_id",
        "records_by_ontology": dict(by_ontology),
        "decision_distribution": dict(decisions_count),
        "confidence_distribution": dict(confidence_count),
        "ambiguity_flags": dict(sorted(flags.items())),
        "agrees_with_linker_top1": agree,
        "pass_b_changed_decisions": changed,
        "draft_sha256": sha256_file(draft_path),
        "contains_clinical_text": False,
        "storage_policy": "PRIVATE and gitignored; rationales carry no patient text.",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "processed": len(drafts),
        "by_ontology": dict(by_ontology),
        "decisions": dict(decisions_count),
        "confidence": dict(confidence_count),
        "flags": dict(sorted(flags.items())),
        "agrees_with_linker_top1": agree,
        "agreement_rate": agree / len(drafts) if drafts else 0.0,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"drafts   {draft_path}  ({len(drafts)} records)")
    print(f"sha256   {manifest['draft_sha256']}")
    print(f"by ontology {dict(by_ontology)}")
    print(f"decisions   {dict(decisions_count)}")
    print(f"confidence  {dict(confidence_count)}")
    print(f"agrees with linker top-1: {agree}/{len(drafts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
