#!/usr/bin/env python3
"""Recover structured RxNorm attributes from the governed local release (Audit 0074 §2-§3).

Streams RXNSAT and RXNREL out of the governed archive without extracting it, keeps only
concepts already inside the frozen competition KB, and records provenance for every field.
Nothing is inferred and no external source is consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.kb.rxnorm.structured import (  # noqa: E402
    ATN_AVAILABLE_STRENGTH,
    ATN_HUMAN_DRUG,
    ATN_QUANTITY,
    ATN_RXTERM_FORM,
    ATN_STRENGTH,
    RECOVERED_ATTRIBUTES,
    RECOVERED_RELATIONS,
    REL_HAS_DOSE_FORM,
    REL_HAS_INGREDIENT,
    REL_HAS_PRECISE_INGREDIENT,
    REL_TRADENAME_OF,
    STRUCTURED_VERSION,
    Provenance,
    StructuredDrug,
    parse_strengths,
)

ARCHIVE = REPO / "data/external/rxnorm/full-2026-07-06/archive/RxNorm_full_07062026.zip"
GOVERNED = REPO / "indices/candidate/rxnorm/competition-v3/index.json"
OUT = REPO / "artifacts/rxnorm_structured/0074"
RELEASE = "rxnorm-full-2026-07-06"

# Verified against the official RxNorm column spec and the governed file itself (Audit 0074).
SAT = {"RXCUI": 0, "ATN": 8, "SAB": 9, "ATV": 10}
REL = {"RXCUI1": 0, "RXCUI2": 4, "RELA": 7, "SAB": 10}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--governed", type=Path, default=GOVERNED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    payload = json.loads(args.governed.read_text(encoding="utf-8"))
    records = {r["concept_id"]: r for r in payload["records"]}
    print(f"governed concepts {len(records):,}")

    archive = zipfile.ZipFile(args.archive)
    attributes: dict[str, dict[str, str]] = defaultdict(dict)
    with archive.open("rrf/RXNSAT.RRF") as handle:
        for line in io.TextIOWrapper(handle, encoding="utf-8", errors="replace"):
            columns = line.split("|")
            if len(columns) <= SAT["ATV"] or columns[SAT["SAB"]] != "RXNORM":
                continue
            rxcui = columns[SAT["RXCUI"]]
            if rxcui not in records:
                continue
            name = columns[SAT["ATN"]]
            if name in RECOVERED_ATTRIBUTES:
                attributes[rxcui].setdefault(name, columns[SAT["ATV"]].strip())
    print(f"concepts with recovered attributes {len(attributes):,}")

    relations: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with archive.open("rrf/RXNREL.RRF") as handle:
        for line in io.TextIOWrapper(handle, encoding="utf-8", errors="replace"):
            columns = line.split("|")
            if len(columns) <= REL["SAB"] or columns[REL["SAB"]] != "RXNORM":
                continue
            rela = columns[REL["RELA"]]
            if rela not in RECOVERED_RELATIONS:
                continue
            # RXNREL is expressed second-to-first: RXCUI2 has the relation to RXCUI1.
            source, target = columns[REL["RXCUI2"]], columns[REL["RXCUI1"]]
            if source in records and target in records:
                relations[source][rela].append(target)
    print(f"concepts with recovered relations  {len(relations):,}")

    def names(ids: list[str], limit: int = 8) -> tuple[str, ...]:
        seen: list[str] = []
        for rxcui in ids:
            label = str(records[rxcui].get("canonical_name", "")).strip()
            if label and label not in seen:
                seen.append(label)
            if len(seen) >= limit:
                break
        return tuple(seen)

    args.out.mkdir(parents=True, exist_ok=True)
    coverage: Counter[str] = Counter()
    by_tty_structured: Counter[str] = Counter()
    by_tty_total: Counter[str] = Counter()
    written = 0

    with (args.out / "structured.jsonl").open("w", encoding="utf-8") as out_handle:
        for rxcui in sorted(records):
            record = records[rxcui]
            metadata = record.get("metadata") or {}
            tty = str(metadata.get("tty", ""))
            by_tty_total[tty] += 1
            attribute = attributes.get(rxcui, {})
            relation = relations.get(rxcui, {})

            provenance: list[Provenance] = []
            for name in sorted(attribute):
                provenance.append(Provenance(RELEASE, "RXNSAT.RRF", name))
            for name in sorted(relation):
                provenance.append(Provenance(RELEASE, "RXNREL.RRF", name))

            strength_text = (
                attribute.get(ATN_STRENGTH) or attribute.get(ATN_AVAILABLE_STRENGTH) or ""
            )
            drug = StructuredDrug(
                rxcui=rxcui,
                tty=tty,
                name=str(record.get("canonical_name", "")),
                ingredients=names(relation.get(REL_HAS_INGREDIENT, [])),
                precise_ingredients=names(relation.get(REL_HAS_PRECISE_INGREDIENT, [])),
                dose_forms=names(relation.get(REL_HAS_DOSE_FORM, [])),
                brands=names(relation.get(REL_TRADENAME_OF, [])),
                strengths=parse_strengths(strength_text),
                available_strength=attribute.get(ATN_AVAILABLE_STRENGTH, ""),
                rxterm_form=attribute.get(ATN_RXTERM_FORM, ""),
                quantity=attribute.get(ATN_QUANTITY, ""),
                human_drug=attribute.get(ATN_HUMAN_DRUG, "").upper() in {"Y", "TRUE", "1"},
                provenance=tuple(provenance),
            )
            if drug.ingredients:
                coverage["ingredient"] += 1
            if drug.precise_ingredients:
                coverage["precise_ingredient"] += 1
            if drug.strengths:
                coverage["strength_parsed"] += 1
            if drug.available_strength:
                coverage["available_strength"] += 1
            if drug.dose_forms:
                coverage["dose_form_relation"] += 1
            if drug.rxterm_form:
                coverage["rxterm_form"] += 1
            if drug.brands:
                coverage["brand"] += 1
            if drug.has_structure:
                coverage["any_structure"] += 1
                by_tty_structured[tty] += 1
            out_handle.write(json.dumps(drug.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            written += 1

    total = len(records)
    manifest: dict[str, Any] = {
        "structured_version": STRUCTURED_VERSION,
        "release": RELEASE,
        "archive": str(args.archive.relative_to(REPO)),
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "governed_index": str(args.governed.relative_to(REPO)),
        "governed_index_sha256": hashlib.sha256(args.governed.read_bytes()).hexdigest(),
        "governed_concepts": total,
        "records_written": written,
        "recovered_attributes": list(RECOVERED_ATTRIBUTES),
        "recovered_relations": list(RECOVERED_RELATIONS),
        "coverage_counts": dict(sorted(coverage.items())),
        "coverage_pct": {k: round(100.0 * v / total, 3) for k, v in sorted(coverage.items())},
        "structured_by_tty": {
            tty: {"total": by_tty_total[tty], "structured": by_tty_structured.get(tty, 0)}
            for tty in sorted(by_tty_total, key=lambda t: -by_tty_total[t])[:14]
        },
        "external_sources_used": [],
        "inferred_fields": [],
        "contains_clinical_text": False,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nwritten {written:,} rows -> {args.out}")
    for key, value in manifest["coverage_pct"].items():
        print(f"  {key:24s} {coverage[key]:>8,}  {value:6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
