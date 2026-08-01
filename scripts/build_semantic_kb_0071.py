#!/usr/bin/env python3
"""Build semantic documents and the concept-disjoint benchmark (Audit 0071 §6, §11A).

Model-free on purpose. This runs on a laptop in seconds and produces everything the GPU
step needs, so the only thing Colab has to do is encode text that has already been reviewed
here. Nothing is downloaded and no network call is made.

Two outputs:

* ``documents.jsonl``  - one bounded semantic document per governed concept. Damaged ICD
  titles are excluded from the embedded text and recorded as provenance instead, because
  embedding `Bao gồm: Bóng` would teach a retriever that an ICD instruction is a concept.
* ``benchmark/{train,dev,test}.jsonl`` - ontology-native positive pairs with typed hard
  negatives, partitioned so that **no concept appears in more than one split**. Splitting by
  record instead of by concept would let a model memorise a concept in training and be
  scored on it again at test time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.kb.icd10.repair.title_recovery import is_damaged_title  # noqa: E402
from mednorm_vi.kb.indexing.normalization import normalize_text  # noqa: E402
from mednorm_vi.linking.semantic.representation import (  # noqa: E402
    REPRESENTATION_VERSION,
    icd_document,
    rxnorm_document,
)

ICD_V41 = REPO / "indices" / "candidate" / "icd10_vi" / "competition-v4.1" / "index.json"
RXNORM = REPO / "indices" / "candidate" / "rxnorm" / "competition-v3" / "index.json"
OUT = REPO / "artifacts" / "semantic" / "0071"

#: Hard-negative kinds, per §11A. Each names *why* the negative is hard, so a failure can be
#: attributed to a mechanism rather than to "the model got it wrong".
NEG_ICD_PARENT = "icd_parent"
NEG_ICD_CHILD = "icd_child"
NEG_ICD_SIBLING = "icd_sibling"
NEG_ICD_OTHER_8 = "icd_other_dot8"
NEG_ICD_UNSPEC_9 = "icd_unspecified_dot9"
NEG_ICD_SAME_ORGAN = "icd_same_organ_different_disease"
NEG_ICD_LEXICAL = "icd_lexically_similar_other_branch"
NEG_ICD_ACCENT = "icd_accent_false_friend"
NEG_RX_TTY = "rxnorm_same_ingredient_different_tty"
NEG_RX_STRENGTH = "rxnorm_same_ingredient_different_strength"
NEG_RX_FORM = "rxnorm_same_ingredient_different_dose_form"
NEG_RX_LEXICAL = "rxnorm_lexical_false_friend"

MAX_NEGATIVES = 8


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["concept_id"]): r for r in payload.get("records", [])}


def strip_accents_key(value: str) -> str:
    return normalize_text(value, strip_accents=True)


def icd_negatives(
    code: str, records: dict[str, dict[str, Any]], by_ascii: dict[str, list[str]]
) -> list[tuple[str, str]]:
    """Typed hard negatives for one ICD concept, deterministic and bounded."""
    out: list[tuple[str, str]] = []
    root = code[:3]
    title = str(records[code].get("canonical_name", ""))

    if code != root and root in records:
        out.append((root, NEG_ICD_PARENT))
    children = sorted(c for c in records if c.startswith(code) and c != code)
    for child in children[:2]:
        kind = (
            NEG_ICD_OTHER_8
            if child.endswith("8")
            else NEG_ICD_UNSPEC_9
            if child.endswith("9")
            else NEG_ICD_CHILD
        )
        out.append((child, kind))
    siblings = sorted(c for c in records if c[:3] == root and c != code and len(c) == len(code))
    for sibling in siblings[:2]:
        out.append((sibling, NEG_ICD_SIBLING))

    # Same chapter letter, different category: same organ system, different disease.
    same_organ = sorted(
        c
        for c in records
        if len(c) == 3
        and c[0] == code[0]
        and c != root
        and not is_damaged_title(str(records[c].get("canonical_name", "")))
    )
    if same_organ:
        out.append((same_organ[len(code) % len(same_organ)], NEG_ICD_SAME_ORGAN))

    # Accent false friend: identical once accents are stripped, different concept.
    for other in by_ascii.get(strip_accents_key(title), []):
        if other != code:
            out.append((other, NEG_ICD_ACCENT))
            break
    return out[:MAX_NEGATIVES]


def rxnorm_negatives(
    cui: str,
    records: dict[str, dict[str, Any]],
    graph: dict[str, list[str]],
    by_ascii: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """Typed hard negatives for one RxCUI, from governed structure only.

    The competition RxNorm index carries `tty` but **not** ingredient, strength, dose form or
    brand as separate fields, so strength/form negatives cannot be built without parsing drug
    names - which would be inventing structure the KB does not assert. Related concepts come
    from the governed graph instead, and the TTY difference is read from metadata that really
    is there. The absent negative kinds are reported rather than faked.
    """
    out: list[tuple[str, str]] = []
    tty = str((records[cui].get("metadata") or {}).get("tty", ""))
    name = str(records[cui].get("canonical_name", ""))

    for other in graph.get(cui, [])[:24]:
        if other == cui or other not in records or len(out) >= MAX_NEGATIVES:
            continue
        other_tty = str((records[other].get("metadata") or {}).get("tty", ""))
        if other_tty and other_tty != tty:
            out.append((other, NEG_RX_TTY))

    for other in by_ascii.get(strip_accents_key(name), []):
        if other != cui and len(out) < MAX_NEGATIVES:
            out.append((other, NEG_RX_LEXICAL))
            break
    return out[:MAX_NEGATIVES]


def build_pairs(
    records: dict[str, dict[str, Any]], ontology: str, negatives: Any
) -> list[dict[str, Any]]:
    """Positive surface forms per concept, each with its typed hard negatives."""
    rows: list[dict[str, Any]] = []
    for code in sorted(records):
        record = records[code]
        title = str(record.get("canonical_name", ""))
        surfaces: list[tuple[str, str]] = []
        if title and not is_damaged_title(title):
            surfaces.append((title, "canonical_title"))
        for alias in record.get("aliases") or []:
            alias = str(alias)
            if alias and alias != title and not is_damaged_title(alias):
                surfaces.append((alias, "governed_alias"))
        if not surfaces:
            continue
        negs = negatives(code)
        for surface, kind in surfaces[:6]:
            rows.append(
                {
                    "query": surface,
                    "positive_concept_id": code,
                    "ontology": ontology,
                    "surface_kind": kind,
                    "hard_negatives": [{"concept_id": c, "negative_kind": k} for c, k in negs],
                }
            )
    return rows


def partition(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Concept-disjoint split. Concepts are assigned, then their rows follow them.

    A concept's hard negatives may name concepts in another split - that is fine and in fact
    necessary, because negatives are scored, never learned as positives.
    """
    concepts = sorted({r["positive_concept_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(concepts)
    n = len(concepts)
    bounds = {
        "train": concepts[: int(n * 0.8)],
        "dev": concepts[int(n * 0.8) : int(n * 0.9)],
        "test": concepts[int(n * 0.9) :],
    }
    assign = {c: split for split, members in bounds.items() for c in members}
    out: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for row in rows:
        out[assign[row["positive_concept_id"]]].append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icd", type=Path, default=ICD_V41)
    parser.add_argument("--rxnorm", type=Path, default=RXNORM)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    icd = load_records(args.icd)
    rx = load_records(args.rxnorm)
    print(f"ICD concepts {len(icd):,} | RxNorm concepts {len(rx):,}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark").mkdir(exist_ok=True)

    # --- semantic documents -------------------------------------------------------------
    parents = {c: str(icd.get(c[:3], {}).get("canonical_name", "")) for c in icd}
    documents = []
    excluded = 0
    for code in sorted(icd):
        doc = icd_document(icd[code], parent_title=parents[code] if len(code) > 3 else "")
        excluded += bool(doc.excluded_damaged_title)
        documents.append(doc.as_dict())
    for cui in sorted(rx):
        documents.append(rxnorm_document(rx[cui]).as_dict())
    with (args.out / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for row in documents:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"documents {len(documents):,} (ICD titles excluded as damaged: {excluded:,})")

    # --- concept-disjoint benchmark ------------------------------------------------------
    by_ascii: dict[str, list[str]] = defaultdict(list)
    for code, record in icd.items():
        title = str(record.get("canonical_name", ""))
        if title and not is_damaged_title(title):
            by_ascii[strip_accents_key(title)].append(code)
    rx_by_ascii: dict[str, list[str]] = defaultdict(list)
    for cui, record in rx.items():
        name = str(record.get("canonical_name", ""))
        if name:
            rx_by_ascii[strip_accents_key(name)].append(cui)
    rx_graph = {
        str(k): [str(x) for x in v]
        for k, v in json.loads(args.rxnorm.read_text(encoding="utf-8")).get("graph", {}).items()
    }

    rows = build_pairs(icd, "ICD10", lambda c: icd_negatives(c, icd, by_ascii))
    rows += build_pairs(rx, "RXNORM", lambda c: rxnorm_negatives(c, rx, rx_graph, rx_by_ascii))
    splits = partition(rows, args.seed)

    concept_sets = {name: {r["positive_concept_id"] for r in part} for name, part in splits.items()}
    overlap = (
        (concept_sets["train"] & concept_sets["dev"])
        | (concept_sets["train"] & concept_sets["test"])
        | (concept_sets["dev"] & concept_sets["test"])
    )
    if overlap:
        raise SystemExit(f"concept leakage across splits: {sorted(overlap)[:5]}")

    negative_kinds: Counter[str] = Counter()
    for part in splits.values():
        for row in part:
            negative_kinds.update(n["negative_kind"] for n in row["hard_negatives"])
    for name, part in splits.items():
        with (args.out / "benchmark" / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in part:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"  {name:6s} rows {len(part):>7,}  concepts {len(concept_sets[name]):>6,}")

    manifest = {
        "representation_version": REPRESENTATION_VERSION,
        "icd_index": str(args.icd.relative_to(REPO)),
        "icd_index_sha256": sha256_file(args.icd),
        "rxnorm_index": str(args.rxnorm.relative_to(REPO)),
        "rxnorm_index_sha256": sha256_file(args.rxnorm),
        "documents": len(documents),
        "icd_damaged_titles_excluded": excluded,
        "benchmark_rows": {k: len(v) for k, v in splits.items()},
        "benchmark_concepts": {k: len(v) for k, v in concept_sets.items()},
        "concept_disjoint": True,
        "concept_leakage": 0,
        "hard_negative_kinds": dict(sorted(negative_kinds.items())),
        "seed": args.seed,
        "deterministic": True,
        "contains_clinical_text": False,
        "note": (
            "Ontology-native supervision only. No public-test text, no clinical records, "
            "no AI-consensus labels are present in this benchmark."
        ),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("\nhard negatives:", dict(sorted(negative_kinds.items())))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
