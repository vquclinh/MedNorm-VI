#!/usr/bin/env python3
"""Export the blind 80-record adjudication packet for a second AI (Audit 0067 §1-§2).

The packet is **blind on purpose**. It carries the mention, its bounded context and every
offered candidate with all locally available structural evidence - and none of Claude's
decisions, confidences or rationales, and no "recommended" candidate. A second opinion that
has already seen the first is not a second opinion; it is a ratification, and the agreement
rate it produces would be meaningless.

`linker_rank` is retained because it is factual metadata about what the production system
returned, but no derived recommendation field is added.

Everything written here lands under `data/private_linker_gold/`, which is gitignored: the
packet contains bounded windows of clinical text.
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

from mednorm_vi.annotation.linker_gold import ONTOLOGY_ICD10, load_records, utc_now  # noqa: E402

BASE = REPO / "data" / "private_linker_gold" / "0065"
PACK = BASE / "pack.jsonl"
QUEUE = BASE / "ai_drafts" / "queues" / "queue_minimum_benchmark.txt"
OUT = BASE / "chatgpt_handoff"

#: Fields that would leak the first opinion. Asserted absent from every exported record.
FORBIDDEN_KEYS = (
    "ai_decision",
    "ai_confidence",
    "rationale",
    "selected_codes",
    "no_valid_candidate",
    "adjudication_status",
    "recommended",
    "recommendation",
    "linker_top1",
    "agrees_with_linker_top1",
    "ambiguity_reason",
)


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


def export_candidate(index: int, candidate: dict[str, Any], ontology: str) -> dict[str, Any]:
    structure = candidate.get("structure", {})
    common = {
        "display_index": index,
        "canonical_name": candidate.get("canonical_name", ""),
        "aliases": candidate.get("aliases", []),
        "retrieval_channels": candidate.get("retrieval_channels", []),
        "lexical_score": candidate.get("score"),
        "linker_rank": candidate.get("linker_rank"),
    }
    if ontology == ONTOLOGY_ICD10:
        return {
            **common,
            "code_dotless": candidate["code"],
            "code_dotted_display": candidate.get("display_code", candidate["code"]),
            "parent_code": structure.get("parent_code"),
            "child_codes": structure.get("child_codes", []),
            "hierarchy_depth": structure.get("hierarchy_depth"),
            "specificity_class": structure.get("specificity_class"),
            "specificity_tier": structure.get("specificity_tier"),
        }
    return {
        **common,
        "rxcui": candidate["code"],
        "tty": structure.get("tty"),
        "ingredient": structure.get("ingredient"),
        "precise_ingredient": structure.get("precise_ingredient"),
        "strength": structure.get("strength"),
        "dose_form": structure.get("dose_form"),
        "brand": structure.get("brand"),
        "graph_relations": structure.get("graph_relations", []),
        "structured_match": structure.get("structured_match", []),
    }


def export_record(record: dict[str, Any]) -> dict[str, Any]:
    ontology = record["ontology"]
    candidates = sorted(record.get("offered_candidates") or [], key=lambda c: c["linker_rank"])
    exported = {
        "annotation_id": record["annotation_id"],
        "ontology": ontology,
        "entity_type": record["entity_type"],
        "source_dataset": record["source_dataset"],
        "stratum": record["provenance"].get("stratum", ""),
        "mention_text": record["mention_text"],
        "context_before": record["context_before"],
        "context_after": record["context_after"],
        "candidate_snapshot_id": record["candidate_snapshot_id"],
        "offered_candidates": [
            export_candidate(i, c, ontology) for i, c in enumerate(candidates, start=1)
        ],
        "offered_count": len(candidates),
    }
    leaked = [k for k in FORBIDDEN_KEYS if k in exported]
    if leaked:
        raise RuntimeError(f"blind packet would leak {leaked} for {record['annotation_id']}")
    return exported


def render_markdown(records: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    lines = [
        "# Blind linker adjudication packet (80 records)",
        "",
        "You are the SECOND independent adjudicator. Another AI has already reviewed these",
        "records; its decisions are deliberately withheld so your judgement is independent.",
        "",
        f"- KB snapshot: `{manifest['kb_snapshot_id']}`",
        f"- ICD index sha256: `{manifest['icd_index_sha256']}`",
        f"- RxNorm index sha256: `{manifest['rxnorm_index_sha256']}`",
        f"- records: {manifest['records']} "
        f"({manifest['records_by_ontology'].get('ICD10', 0)} ICD-10, "
        f"{manifest['records_by_ontology'].get('RXNORM', 0)} RxNorm)",
        "",
        "## Task",
        "",
        "For each record choose `SELECTED`, `NO_VALID_CANDIDATE` or `ABSTAIN`.",
        "",
        "* Use ONLY the mention, its context and the candidate evidence shown. Do not invent",
        "  a code that is not listed.",
        "* `SELECTED` requires the context to support the specific concept. Do not infer an",
        "  organ, cause, acuity, complication, strength or dose form the context omits.",
        "* `NO_VALID_CANDIDATE` when the mention is an umbrella phrase, a drug class, a brand",
        "  absent from the knowledge base, or when none of the candidates matches.",
        "* `ABSTAIN` when several candidates stay plausible and the context cannot separate them.",
        "* ICD codes must be returned DOTLESS (`I269`, not `I26.9`). RxNorm returns the RxCUI.",
        "* Multiple codes are allowed only when the mention genuinely maps to more than one.",
        "",
        "Return one JSON object per record, matching `ai_second_opinion_v1.schema.json`:",
        "",
        "```json",
        '{"annotation_id":"lg1-...","ontology":"ICD10","decision":"SELECTED",',
        ' "selected_codes":["I269"],"confidence":"HIGH","rationale":"...",',
        ' "evidence_used":["mention_text","context_after","specificity_class"],',
        ' "competing_codes":["I509"],"ambiguity_flags":["parent_child_ambiguity"]}',
        "```",
        "",
        "Write results to `chatgpt_decisions.jsonl`, one object per line, 80 lines.",
        "",
        "---",
        "",
    ]
    for position, record in enumerate(records, start=1):
        lines += [
            f"## {position}. `{record['annotation_id']}` — {record['ontology']}",
            "",
            f"- stratum: `{record['stratum']}` · source: `{record['source_dataset']}`",
            f"- **mention: `{record['mention_text']}`**",
            "",
            "```text",
            f"...{record['context_before']}",
            f"    >>> {record['mention_text']} <<<",
            f"{record['context_after']}...",
            "```",
            "",
        ]
        if not record["offered_candidates"]:
            lines += [
                "**No candidates were offered for this mention.** This is itself a finding:",
                "answer `NO_VALID_CANDIDATE` unless you can justify otherwise from the evidence.",
                "",
            ]
            continue
        if record["ontology"] == ONTOLOGY_ICD10:
            lines.append("| # | code | dotted | name | class | depth | parent | channels | score |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for c in record["offered_candidates"]:
                lines.append(
                    f"| {c['display_index']} | `{c['code_dotless']}` "
                    f"| `{c['code_dotted_display']}` "
                    f"| {c['canonical_name'][:52]} | {c['specificity_class']} "
                    f"| {c['hierarchy_depth']} | `{c['parent_code'] or '-'}` "
                    f"| {', '.join(c['retrieval_channels']) or '-'} | {c['lexical_score']:.2f} |"
                )
        else:
            lines.append(
                "| # | RxCUI | name | TTY | ingredient | strength | form | brand "
                "| channels | score |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for c in record["offered_candidates"]:
                lines.append(
                    f"| {c['display_index']} | `{c['rxcui']}` "
                    f"| {c['canonical_name'][:44]} "
                    f"| {c['tty'] or '-'} | {c['ingredient'] or '-'} | {c['strength'] or '-'} "
                    f"| {c['dose_form'] or '-'} | {c['brand'] or '-'} "
                    f"| {', '.join(c['retrieval_channels']) or '-'} | {c['lexical_score']:.2f} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    ids = [
        line.strip() for line in args.queue.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    by_id = {r["annotation_id"]: r for r in load_records(args.pack)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"queue references {len(missing)} unknown record(s): {missing[:3]}")

    records = [export_record(by_id[i]) for i in sorted(ids)]
    args.out.mkdir(parents=True, exist_ok=True)

    jsonl = args.out / "minimum_benchmark_blind.jsonl"
    jsonl.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
        encoding="utf-8",
    )
    by_ontology = Counter(r["ontology"] for r in records)
    manifest: dict[str, Any] = {
        "packet": "minimum_benchmark_blind",
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "blind": True,
        "blind_note": (
            "Contains NO first-opinion decision, confidence, rationale or recommended "
            "candidate. linker_rank is retained as factual metadata only."
        ),
        "records": len(records),
        "records_by_ontology": dict(by_ontology),
        "expected_annotation_ids": sorted(ids),
        "kb_snapshot_id": records[0]["candidate_snapshot_id"] if records else "",
        "pack_sha256": sha256_file(args.pack),
        "icd_index_sha256": sha256_file(
            REPO / "indices" / "candidate" / "icd10_vi" / "competition-v3" / "index.json"
        ),
        "rxnorm_index_sha256": sha256_file(
            REPO / "indices" / "candidate" / "rxnorm" / "competition-v3" / "index.json"
        ),
        "response_schema": "schemas/annotation/ai_second_opinion_v1.schema.json",
        "expected_response_file": "chatgpt_decisions.jsonl",
        "contains_clinical_text": True,
        "storage_policy": "PRIVATE; data/** is gitignored and this packet must never be tracked.",
    }
    markdown = args.out / "minimum_benchmark_blind.md"
    markdown.write_text(render_markdown(records, manifest), encoding="utf-8")

    manifest["blind_jsonl_sha256"] = sha256_file(jsonl)
    manifest["blind_markdown_sha256"] = sha256_file(markdown)
    (args.out / "handoff_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"jsonl    {jsonl}\n         sha256 {manifest['blind_jsonl_sha256']}")
    print(f"markdown {markdown}\n         sha256 {manifest['blind_markdown_sha256']}")
    print(f"records  {len(records)}  {dict(by_ontology)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
