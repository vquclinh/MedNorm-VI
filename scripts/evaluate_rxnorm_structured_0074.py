#!/usr/bin/env python3
"""Governed A/B ablation for structured RxNorm evidence (Audit 0074 §6).

    A = frozen 0073 RxNorm S1 candidate ordering (lexical retrieval order)
    B = A + structured compatibility reordering

Ontology-native only: queries are built from the governed source's own recorded structure.
No public-leaderboard signal and no clinical text are used anywhere.

The dense and reranker stages need a GPU, so this measures the stages that are measurable
locally - retrieval coverage and the structured reordering - and says so rather than implying
a complete-system result.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.kb.indexing.retrieval import load_index, search_index  # noqa: E402
from mednorm_vi.kb.rxnorm.structured import StructuredDrug  # noqa: E402
from mednorm_vi.linking.semantic import rxnorm_structured as rs  # noqa: E402

STRUCTURED = REPO / "artifacts/rxnorm_structured/0074/structured.jsonl"
RXNORM = REPO / "indices/candidate/rxnorm/competition-v3/index.json"
OUT = REPO / "artifacts/rxnorm_structured/0074"
TOPK = 20


def load_structured(path: Path) -> dict[str, StructuredDrug]:
    out: dict[str, StructuredDrug] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                drug = StructuredDrug.from_dict(json.loads(line))
                out[drug.rxcui] = drug
    return out


VI_FORM = {
    "tablet": "viên nén",
    "capsule": "viên nang",
    "injection": "ống tiêm",
    "solution": "dung dịch",
    "suspension": "hỗn dịch",
    "syrup": "siro",
    "cream": "kem",
    "ointment": "thuốc mỡ",
    "patch": "miếng dán",
}


def vietnamese_form(drug: StructuredDrug) -> str:
    """The Vietnamese word a clinician would use for this concept's recorded dose form."""
    recorded = " ".join((*drug.dose_forms, drug.rxterm_form)).casefold()
    for english, vietnamese in VI_FORM.items():
        if english in recorded:
            return vietnamese
    return ""


def build_cases(
    structured: dict[str, StructuredDrug], seed: int, limit: int
) -> list[dict[str, Any]]:
    """Ontology-native cases rendered the way a Vietnamese clinical note writes them.

    Every case restates facts the governed source already records - ingredient, strength,
    dose form - in the surface form a clinician would type. Nothing new is asserted. The
    first benchmark built queries verbatim from each concept's own fields, which lexical
    retrieval solved at R@1 0.957 with hard-negative rates near zero; it measured string
    copying rather than drug discrimination.

    Four disjoint variants, so the §6 subsets actually separate:

        strength_compact   `amoxicillin 500mg`      - no space, as notes are written
        vi_form            `amoxicillin viên nén`   - Vietnamese form, NO strength
        strength_vi_form   `amoxicillin 500mg viên nén`
        ingredient_only    `amoxicillin`            - ambiguous by construction
    """
    by_ingredient: dict[str, list[str]] = {}
    for drug in structured.values():
        if drug.ingredients and drug.tty in {"SCD", "SBD"}:
            by_ingredient.setdefault(drug.ingredients[0].casefold(), []).append(drug.rxcui)

    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    variants = ("strength_compact", "vi_form", "strength_vi_form", "ingredient_only")
    for position, ingredient in enumerate(sorted(by_ingredient)):
        siblings = sorted(by_ingredient[ingredient])
        if len(siblings) < 3:
            continue  # need real competition inside the family
        gold = structured[siblings[0]]
        name = gold.ingredients[0]
        strength = gold.strengths[0] if gold.strengths else None
        form = vietnamese_form(gold)
        variant = variants[position % len(variants)]

        if variant == "strength_compact" and strength:
            query = f"{name} {strength.value}{strength.unit}"
        elif variant == "vi_form" and form:
            query = f"{name} {form}"
        elif variant == "strength_vi_form" and strength and form:
            query = f"{name} {strength.value}{strength.unit} {form}"
        elif variant == "ingredient_only":
            query = name
        else:
            continue

        wrong_strength = [
            c
            for c in siblings[1:]
            if structured[c].strengths
            and gold.strengths
            and not (structured[c].strength_keys & gold.strength_keys)
        ][:5]
        wrong_form = [
            c
            for c in siblings[1:]
            if vietnamese_form(structured[c]) and form and vietnamese_form(structured[c]) != form
        ][:5]
        cases.append(
            {
                "query": query,
                "variant": variant,
                "gold": gold.rxcui,
                "states_strength": bool(strength)
                and variant != "vi_form"
                and variant != "ingredient_only",
                "states_form": bool(form) and variant in {"vi_form", "strength_vi_form"},
                "wrong_strength_siblings": wrong_strength,
                "wrong_form_siblings": wrong_form,
                "brand": bool(gold.brands),
            }
        )
        if len(cases) >= limit:
            break
    rng.shuffle(cases)
    return cases


def metrics(results: dict[str, list[str]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    at = {k: 0 for k in (1, 2, 5, 10)}
    reciprocal = 0.0
    covered = 0
    for case in cases:
        ranked = results[case["query"]]
        gold = case["gold"]
        if gold in ranked:
            covered += 1
            reciprocal += 1.0 / (ranked.index(gold) + 1)
        for k in at:
            if gold in ranked[:k]:
                at[k] += 1
    total = len(cases) or 1
    return {
        **{f"recall_at_{k}": round(at[k] / total, 4) for k in at},
        "mrr": round(reciprocal / total, 4),
        "retrieval_coverage": round(covered / total, 4),
        "n": len(cases),
    }


def subset(results: dict[str, list[str]], cases: list[dict[str, Any]], key: str) -> dict[str, Any]:
    chosen = [c for c in cases if c.get(key)]
    return metrics(results, chosen) if chosen else {"n": 0}


def hard_negative_rate(
    results: dict[str, list[str]], cases: list[dict[str, Any]], key: str
) -> dict[str, Any]:
    """How often a same-ingredient/wrong-facet sibling outranks the gold concept."""
    above = considered = 0
    for case in cases:
        siblings = [s for s in case[key] if s]
        if not siblings:
            continue
        ranked = results[case["query"]]
        if case["gold"] not in ranked:
            continue
        considered += 1
        gold_at = ranked.index(case["gold"])
        if any(s in ranked and ranked.index(s) < gold_at for s in siblings):
            above += 1
    return {
        "considered": considered,
        "hard_negative_above_gold": above,
        "rate": round(above / considered, 4) if considered else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured", type=Path, default=STRUCTURED)
    parser.add_argument("--rxnorm", type=Path, default=RXNORM)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    structured = load_structured(args.structured)
    index = load_index(args.rxnorm)
    cases = build_cases(structured, args.seed, args.limit)
    print(f"structured concepts {len(structured):,} | benchmark cases {len(cases):,}")

    a: dict[str, list[str]] = {}
    b: dict[str, list[str]] = {}
    for case in cases:
        ranked = [h.concept_id for h in search_index(index, case["query"], limit=TOPK)]
        a[case["query"]] = ranked
        b[case["query"]] = rs.reorder(ranked, case["query"], structured)

    fixes = regressions = 0
    for case in cases:
        gold = case["gold"]
        before, after = a[case["query"]], b[case["query"]]
        hit_before = before[:1] == [gold]
        hit_after = after[:1] == [gold]
        fixes += hit_after and not hit_before
        regressions += hit_before and not hit_after

    report = {
        "note": "Retrieval + structured reordering only. Dense and reranker stages require a "
        "GPU and are NOT included; this is not a complete-system result.",
        "topk": TOPK,
        "A_frozen_0073_rxnorm_s1": metrics(a, cases),
        "B_plus_structured": metrics(b, cases),
        "subsets": {
            "exact_strength_stated": {
                "A": subset(a, cases, "states_strength"),
                "B": subset(b, cases, "states_strength"),
            },
            "form_stated": {
                "A": subset(a, cases, "states_form"),
                "B": subset(b, cases, "states_form"),
            },
            "brand_present": {"A": subset(a, cases, "brand"), "B": subset(b, cases, "brand")},
        },
        "hard_negatives": {
            "same_ingredient_wrong_strength": {
                "A": hard_negative_rate(a, cases, "wrong_strength_siblings"),
                "B": hard_negative_rate(b, cases, "wrong_strength_siblings"),
            },
            "same_ingredient_wrong_form": {
                "A": hard_negative_rate(a, cases, "wrong_form_siblings"),
                "B": hard_negative_rate(b, cases, "wrong_form_siblings"),
            },
        },
        "by_variant": {
            v: {
                "A": metrics(a, [c for c in cases if c["variant"] == v]),
                "B": metrics(b, [c for c in cases if c["variant"] == v]),
            }
            for v in sorted({c["variant"] for c in cases})
        },
        "ranking_fixes_top1": fixes,
        "ranking_regressions_top1": regressions,
        "candidate_set_preserved": all(sorted(a[q]) == sorted(b[q]) for q in a),
        "null_policy": "shadow (unchanged); no candidate is removed by this milestone",
        "public_leaderboard_signal_used": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    # Freeze the exact case set so the GPU complete-system A/B runs the SAME queries.
    (args.out / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    (args.out / "ab_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
