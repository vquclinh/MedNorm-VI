#!/usr/bin/env python3
"""Build ICD KB competition-v4.1: wrapped-title recovery + tiered evidence (Audit 0069 §7).

competition-v3 and competition-v4 are both immutable references; v4.1 is a new versioned
artifact beside them, so rollback stays a config edit. Two things change relative to v4:

* titles are recovered with x-aligned row reconstruction, so a title split across physical
  lines is rejoined instead of stored truncated (`Thiếu máu do thiếu` -> `… vitamin B12`);
* postings are split by evidence kind, so ranking can be lexicographic over evidence tiers
  instead of additive over a single score. That is what stops 29,586 aliases' worth of
  trigram noise from outranking an exact match, which is how v4 lost top-1 accuracy.

The concept set is identical to v3 and v4: no concept is added, removed or re-identified,
and every governance field is carried across untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.kb.icd10.repair.alias_layer import (  # noqa: E402
    ALIAS_CANONICAL,
    ALIAS_SYNONYM_RULE,
    ALIAS_UNACCENTED,
    EVIDENCE_RECALL_ONLY,
    REJECTED_RULES,
    SYNONYM_RULES,
    aliases_for,
)
from mednorm_vi.kb.icd10.repair.title_recovery import (  # noqa: E402
    MAX_JOINED_LINES,
    is_damaged_title,
    recover_titles_v2,
    trim_embedded_note,
)
from mednorm_vi.kb.indexing.evidence import policy_document  # noqa: E402
from mednorm_vi.kb.indexing.normalization import (  # noqa: E402
    accent_marked_ngrams,
    accent_marked_tokens,
    char_ngrams,
    normalize_text,
    tokens,
)

V3 = REPO / "indices" / "candidate" / "icd10_vi" / "competition-v3" / "index.json"
OUT = REPO / "indices" / "candidate" / "icd10_vi" / "competition-v4.1"

#: An accented alias shared by this many codes is a generic fragment rather than a name, so
#: it is recorded but kept out of the exact tiers, where it could only produce ties. Its
#: provenance is never deleted - it stays in the alias manifest, flagged.
COLLISION_DEMOTION_THRESHOLD = 10


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--pdf-text", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    payload = json.loads(args.v3.read_text(encoding="utf-8"))
    records = {r["concept_id"]: r for r in payload["records"]}
    damaged = {
        c: r.get("canonical_name", "")
        for c, r in records.items()
        if is_damaged_title(r.get("canonical_name", ""))
    }
    pdf_sha = sha256_bytes(args.pdf_text.read_bytes())
    lines = args.pdf_text.read_text(encoding="utf-8").splitlines()
    recovered, unrecovered = recover_titles_v2(lines, damaged, source_sha256=pdf_sha)
    print(f"v3 concepts {len(records):,} | damaged {len(damaged):,} "
          f"| recovered {len(recovered):,} | still damaged {len(unrecovered):,}")

    # A title can carry a note without starting with one - I25 is stored as
    # `Bệnh tim thiếu máu cục bộ Loại trừ: bệnh`, which `is_damaged_title` cannot see because
    # it anchors at the start. Trimming the glued note is deterministic and reversible (the
    # previous string is kept in the repair manifest), and it is what makes the concept
    # reachable by its own name.
    trimmed: dict[str, tuple[str, str]] = {}
    titles: dict[str, str] = {}
    for concept_id in sorted(records):
        if concept_id in recovered:
            titles[concept_id] = recovered[concept_id].recovered_title
            continue
        current = records[concept_id].get("canonical_name", "")
        clean = trim_embedded_note(current)
        if clean and clean != current and not is_damaged_title(clean):
            trimmed[concept_id] = (current, clean)
            titles[concept_id] = clean
        else:
            titles[concept_id] = current
    print(f"embedded-note trims {len(trimmed):,}")

    # --- alias pass one: generate, then measure collisions before deciding tiers ---
    per_concept: dict[str, list[Any]] = {}
    for concept_id in sorted(records):
        per_concept[concept_id] = aliases_for(concept_id, titles[concept_id], source="tt06-2026")

    collisions: dict[str, set[str]] = defaultdict(set)
    for concept_id, aliases in per_concept.items():
        for alias in aliases:
            collisions[normalize_text(alias.alias)].add(concept_id)
    demoted = {k for k, v in collisions.items() if len(v) >= COLLISION_DEMOTION_THRESHOLD}

    # --- pass two: postings, split by evidence kind so ranking can be lexicographic ---
    exact: dict[str, set[str]] = defaultdict(set)
    exact_canonical: dict[str, set[str]] = defaultdict(set)
    exact_alias: dict[str, set[str]] = defaultdict(set)
    exact_synonym: dict[str, set[str]] = defaultdict(set)
    exact_ascii: dict[str, set[str]] = defaultdict(set)
    ngrams: dict[str, set[str]] = defaultdict(set)
    ngrams_accent: dict[str, set[str]] = defaultdict(set)
    sparse: dict[str, set[str]] = defaultdict(set)
    sparse_accent: dict[str, set[str]] = defaultdict(set)

    out_records: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    alias_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()

    for concept_id in sorted(records):
        record = dict(records[concept_id])
        repaired = recovered.get(concept_id)
        if repaired is not None:
            record["canonical_name"] = repaired.recovered_title
            metadata = dict(record.get("metadata") or {})
            metadata["title_repaired"] = "true"
            metadata["title_source"] = "tt06-2026-vietnamese-title-column"
            metadata["title_recovery_confidence"] = repaired.confidence
            metadata["title_source_lines"] = ",".join(str(n) for n in repaired.source_lines)
            record["metadata"] = metadata
        elif concept_id in trimmed:
            record["canonical_name"] = trimmed[concept_id][1]
            metadata = dict(record.get("metadata") or {})
            metadata["title_repaired"] = "true"
            metadata["title_source"] = "embedded_note_trim"
            metadata["title_previous"] = trimmed[concept_id][0]
            record["metadata"] = metadata

        governed = per_concept[concept_id]
        record["aliases"] = [a.alias for a in governed]
        alias_counts.update(a.alias_type for a in governed)
        class_counts.update(a.evidence_class for a in governed)

        for alias in governed:
            key = normalize_text(alias.alias)
            ascii_key = normalize_text(alias.alias, strip_accents=True)
            row = alias.as_dict()
            row["collision_codes"] = len(collisions[key])
            row["demoted_from_exact_tier"] = key in demoted
            alias_rows.append(row)

            # Accent-stripped forms are recall-only: they feed the accent-insensitive
            # channels and nothing else, so they can never reach tiers A-D.
            if alias.evidence_class == EVIDENCE_RECALL_ONLY:
                exact_ascii[ascii_key].add(concept_id)
            else:
                exact[key].add(concept_id)
                exact_ascii[ascii_key].add(concept_id)
                if key not in demoted:
                    if alias.alias_type == ALIAS_SYNONYM_RULE:
                        exact_synonym[key].add(concept_id)
                    elif alias.alias_type == ALIAS_CANONICAL:
                        exact_canonical[key].add(concept_id)
                    else:
                        exact_alias[key].add(concept_id)
                for gram in accent_marked_ngrams(alias.alias):
                    ngrams_accent[gram].add(concept_id)
                for token in accent_marked_tokens(alias.alias):
                    sparse_accent[token].add(concept_id)

            for gram in char_ngrams(alias.alias):
                ngrams[gram].add(concept_id)
            for token in tokens(alias.alias):
                sparse[token].add(concept_id)

        # Codes stay searchable exactly as in v3, and rank as canonical evidence.
        dotted = str((record.get("metadata") or {}).get("dotted_code", "")) or concept_id
        for form in {dotted, concept_id}:
            key = normalize_text(form)
            exact[key].add(concept_id)
            exact_canonical[key].add(concept_id)
            exact_ascii[normalize_text(form, strip_accents=True)].add(concept_id)
            for gram in char_ngrams(form):
                ngrams[gram].add(concept_id)
            for gram in accent_marked_ngrams(form):
                ngrams_accent[gram].add(concept_id)
            for token in tokens(form):
                sparse[token].add(concept_id)
            for token in accent_marked_tokens(form):
                sparse_accent[token].add(concept_id)
        out_records.append(record)

    def freeze(mapping: dict[str, set[str]]) -> dict[str, list[str]]:
        return {k: sorted(v) for k, v in sorted(mapping.items())}

    index = {
        "metadata": {
            **payload.get("metadata", {}),
            "index_id": "icd10_vi-competition-index-v4.1",
            "source_snapshot_id": "competition-kb-v4.1-icd-repaired-tiered",
            "record_count": len(out_records),
            "concept_count": len(out_records),
            "repaired_titles": len(recovered),
            "remaining_damaged_titles": len(unrecovered),
            "alias_count": sum(len(r["aliases"]) for r in out_records),
            "retrieval_policy_id": policy_document()["policy_id"],
            "derived_from": "competition-v3 + wrapped-title recovery + tiered alias layer",
            "builder_version": "kb-index-v1+icd-repair-v4.1",
        },
        "records": out_records,
        "exact": freeze(exact),
        "exact_canonical": freeze(exact_canonical),
        "exact_alias": freeze(exact_alias),
        "exact_synonym": freeze(exact_synonym),
        "exact_ascii": freeze(exact_ascii),
        "ngrams": freeze(ngrams),
        "ngrams_accent": freeze(ngrams_accent),
        "sparse_terms": freeze(sparse),
        "sparse_accent": freeze(sparse_accent),
        "graph": payload.get("graph", {}),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    index_sha = sha256_bytes((args.out / "index.json").read_bytes())

    shared = Counter(len(v) for v in collisions.values())
    accented = {k: v for k, v in collisions.items() if any(
        a.alias_type != ALIAS_UNACCENTED
        for c in v for a in per_concept[c] if normalize_text(a.alias) == k
    )}
    collision_report = {
        "normalized_aliases": len(collisions),
        "unique_to_one_code": shared.get(1, 0),
        "shared_by_2_codes": shared.get(2, 0),
        "shared_by_3_or_more": sum(n for size, n in shared.items() if size >= 3),
        "max_codes_per_alias": max(shared) if shared else 0,
        "demotion_threshold": COLLISION_DEMOTION_THRESHOLD,
        "demoted_from_exact_tier": len(demoted),
        "accented_alias_keys": len(accented),
        "unaccented_collision_keys": len(collisions) - len(accented),
        "high_frequency_ambiguous": [
            {"alias": k, "codes": len(v)}
            for k, v in sorted(collisions.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:40]
        ],
    }
    (args.out / "alias_collision_report.json").write_text(
        json.dumps(collision_report, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    (args.out / "retrieval_policy.json").write_text(
        json.dumps(policy_document(), ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    (args.out / "repair_manifest.json").write_text(
        json.dumps(
            {
                "method": "x_aligned_row_reconstruction_then_column_extraction",
                "max_joined_lines": MAX_JOINED_LINES,
                "source_sha256": pdf_sha,
                "damaged_before": len(damaged),
                "repaired_count": len(recovered),
                "damaged_after": len(unrecovered),
                "embedded_note_trims": len(trimmed),
                "trimmed": {
                    c: {"previous": a, "repaired": b}
                    for c, (a, b) in sorted(trimmed.items())
                },
                "by_confidence": dict(Counter(t.confidence for t in recovered.values())),
                "wrapped_recoveries": sum(
                    1 for t in recovered.values() if t.joined_line_count > 1
                ),
                "category_codes": sum(1 for c in recovered if len(c) == 3),
                "specific_codes": sum(1 for c in recovered if len(c) > 3),
                "repaired": {c: v.as_dict() for c, v in sorted(recovered.items())},
                "unrecovered": dict(sorted(unrecovered.items())),
            },
            ensure_ascii=False, indent=1, sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.out / "alias_manifest.json").write_text(
        json.dumps(
            {
                "alias_rows": len(alias_rows),
                "by_type": dict(alias_counts),
                "by_evidence_class": dict(class_counts),
                "synonym_rules_applied": [r for r in SYNONYM_RULES if r[2]],
                "synonym_rules_rejected": list(REJECTED_RULES),
                "aliases": alias_rows,
            },
            ensure_ascii=False, indent=1, sort_keys=True,
        ),
        encoding="utf-8",
    )
    diagnostics = {
        "concept_count": len(out_records),
        "repaired_titles": len(recovered),
        "embedded_note_trims": len(trimmed),
        "remaining_damaged_titles": len(unrecovered),
        "alias_count": sum(len(r["aliases"]) for r in out_records),
        "alias_by_type": dict(alias_counts),
        "alias_by_evidence_class": dict(class_counts),
        "postings": {
            "exact": len(exact), "exact_canonical": len(exact_canonical),
            "exact_alias": len(exact_alias), "exact_synonym": len(exact_synonym),
            "exact_ascii": len(exact_ascii), "ngrams": len(ngrams),
            "ngrams_accent": len(ngrams_accent), "sparse_terms": len(sparse),
            "sparse_accent": len(sparse_accent),
        },
        "collisions": {
            k: v for k, v in collision_report.items() if k != "high_frequency_ambiguous"
        },
        "index_sha256": index_sha,
    }
    (args.out / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "version": "competition-v4.1",
                "derived_from_index": str(args.v3.relative_to(REPO)),
                "derived_from_sha256": sha256_bytes(args.v3.read_bytes()),
                "pdf_text_sha256": pdf_sha,
                "governed_source": "data/external/icd10_vi/tt06-2026-official/06-byt-kem.pdf",
                "governed_source_sha256": (
                    "8639f5eeb77b571363dc841923095895d2498748f9bc6620f50710a6da9159e2"
                ),
                "record_count": len(out_records),
                "repaired_titles": len(recovered),
                "alias_count": sum(len(r["aliases"]) for r in out_records),
                "retrieval_policy_id": policy_document()["policy_id"],
                "index_sha256": index_sha,
                "deterministic": True,
                "seed": None,
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nv4.1 written to {args.out}")
    for key, value in diagnostics["postings"].items():
        print(f"  postings {key:18s} {value:,}")
    print(f"  aliases        {diagnostics['alias_count']:,}  {dict(class_counts)}")
    print(f"  collisions     unique={collision_report['unique_to_one_code']:,} "
          f"2={collision_report['shared_by_2_codes']:,} "
          f">=3={collision_report['shared_by_3_or_more']:,} "
          f"demoted={len(demoted):,}")
    print(f"  index sha256   {index_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
