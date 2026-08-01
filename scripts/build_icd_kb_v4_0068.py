#!/usr/bin/env python3
"""Build ICD KB competition-v4: repaired titles + governed alias layer (Audit 0068 §7).

competition-v3 is never mutated; v4 is a new versioned artifact beside it, so rollback is
a config edit. Only two things change: damaged canonical names are replaced by titles
recovered from the governed TT06-2026 source, and an alias layer is populated where v3 had
none. Every governance field (membership, runtime_role, name_quality, quality_flags) is
carried across untouched, and the concept set is identical - no concept is added, removed
or re-identified.

Postings are rebuilt with the same primitives the governed builder uses, so the retrieval
semantics of v4 are the v3 semantics applied to repaired text.
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
    REJECTED_RULES,
    SYNONYM_RULES,
    aliases_for,
)
from mednorm_vi.kb.icd10.repair.title_recovery import (  # noqa: E402
    is_damaged_title,
    recover_titles,
)
from mednorm_vi.kb.indexing.normalization import char_ngrams, normalize_text, tokens  # noqa: E402

V3 = REPO / "indices" / "candidate" / "icd10_vi" / "competition-v3" / "index.json"
V4 = REPO / "indices" / "candidate" / "icd10_vi" / "competition-v4"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--pdf-text", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=V4)
    args = parser.parse_args(argv)

    payload = json.loads(args.v3.read_text(encoding="utf-8"))
    records = {r["concept_id"]: r for r in payload["records"]}
    damaged = {
        c: r.get("canonical_name", "")
        for c, r in records.items()
        if is_damaged_title(r.get("canonical_name", ""))
    }
    print(f"v3 concepts {len(records):,} | damaged titles {len(damaged):,}")

    lines = args.pdf_text.read_text(encoding="utf-8").splitlines()
    recovered, unrecovered = recover_titles(lines, damaged)
    print(f"recovered {len(recovered):,} | still damaged {len(unrecovered):,}")

    exact: dict[str, set[str]] = defaultdict(set)
    exact_ascii: dict[str, set[str]] = defaultdict(set)
    ngrams: dict[str, set[str]] = defaultdict(set)
    sparse: dict[str, set[str]] = defaultdict(set)

    def add(concept_id: str, alias: str) -> None:
        if not alias.strip():
            return
        exact[normalize_text(alias)].add(concept_id)
        exact_ascii[normalize_text(alias, strip_accents=True)].add(concept_id)
        for gram in char_ngrams(alias):
            ngrams[gram].add(concept_id)
        for token in tokens(alias):
            sparse[token].add(concept_id)

    out_records: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    alias_counts: Counter[str] = Counter()
    for concept_id in sorted(records):
        record = dict(records[concept_id])
        repaired = recovered.get(concept_id)
        if repaired is not None:
            record["canonical_name"] = repaired.recovered_title
            metadata = dict(record.get("metadata") or {})
            metadata["title_repaired"] = "true"
            metadata["title_source"] = "tt06-2026-vietnamese-title-column"
            record["metadata"] = metadata
        title = record.get("canonical_name", "")

        governed = aliases_for(concept_id, title, source="tt06-2026")
        record["aliases"] = [a.alias for a in governed]
        alias_counts.update(a.alias_type for a in governed)
        alias_rows.extend(a.as_dict() for a in governed)

        # Codes stay searchable exactly as in v3.
        dotted = str((record.get("metadata") or {}).get("dotted_code", "")) or concept_id
        for alias in {*(a.alias for a in governed), dotted, concept_id}:
            add(concept_id, alias)
        out_records.append(record)

    index = {
        "metadata": {
            **payload.get("metadata", {}),
            "index_id": "icd10_vi-competition-index-v4",
            "source_snapshot_id": "competition-kb-v4-icd-repaired",
            "record_count": len(out_records),
            "concept_count": len(out_records),
            "repaired_titles": len(recovered),
            "alias_count": sum(len(r["aliases"]) for r in out_records),
            "derived_from": "competition-v3 + TT06-2026 title recovery + governed alias layer",
            "builder_version": "kb-index-v1+icd-repair-v4",
        },
        "records": out_records,
        "exact": {k: sorted(v) for k, v in sorted(exact.items())},
        "exact_ascii": {k: sorted(v) for k, v in sorted(exact_ascii.items())},
        "ngrams": {k: sorted(v) for k, v in sorted(ngrams.items())},
        "sparse_terms": {k: sorted(v) for k, v in sorted(sparse.items())},
        "graph": payload.get("graph", {}),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    body = json.dumps(index, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = sha256_bytes(body)
    index["metadata"]["deterministic_index_hash"] = digest
    (args.out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    (args.out / "repair_manifest.json").write_text(
        json.dumps(
            {
                "repaired": {c: v.as_dict() for c, v in sorted(recovered.items())},
                "unrecovered": dict(sorted(unrecovered.items())),
                "damaged_before": len(damaged),
                "repaired_count": len(recovered),
                "damaged_after": len(unrecovered),
            },
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.out / "alias_manifest.json").write_text(
        json.dumps(
            {
                "alias_rows": len(alias_rows),
                "by_type": dict(alias_counts),
                "synonym_rules_applied": [r for r in SYNONYM_RULES if r[2]],
                "synonym_rules_rejected": list(REJECTED_RULES),
                "aliases": alias_rows[:50000],
            },
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "version": "competition-v4",
                "derived_from_index": str(args.v3.relative_to(REPO)),
                "derived_from_sha256": sha256_bytes(args.v3.read_bytes()),
                "pdf_text_sha256": sha256_bytes(args.pdf_text.read_bytes()),
                "governed_source": "data/external/icd10_vi/tt06-2026-official/06-byt-kem.pdf",
                "governed_source_sha256": (
                    "8639f5eeb77b571363dc841923095895d2498748f9bc6620f50710a6da9159e2"
                ),
                "record_count": len(out_records),
                "repaired_titles": len(recovered),
                "alias_count": sum(len(r["aliases"]) for r in out_records),
                "index_sha256": sha256_bytes((args.out / "index.json").read_bytes()),
                "deterministic": True,
                "seed": None,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nv4 written to {args.out}")
    print(f"  records        {len(out_records):,}")
    print(f"  repaired       {len(recovered):,}")
    print(f"  aliases        {sum(len(r['aliases']) for r in out_records):,}  {dict(alias_counts)}")
    print(f"  exact postings {len(exact):,}  (v3 had {len(payload.get('exact', {})):,})")
    print(f"  index sha256   {sha256_bytes((args.out / 'index.json').read_bytes())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
