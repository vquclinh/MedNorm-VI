#!/usr/bin/env python3
"""Build the stratified private linker-gold annotation pack (Audit 0065 §4-§5, §7).

Audit 0063 §7.1 established that no ICD or RxNorm gold codes exist anywhere in this
project, which left retrieval Recall@k and MRR undefined. This script assembles the
evidence a human needs to create the first honest ones.

Design decisions worth stating, because each is a way the benchmark could have been
quietly biased:

* **Sampling uses GOLD mention spans and types only.** E3 predictions never decide
  eligibility. If predictions chose the sample, the benchmark would measure the linker on
  the mentions the mention layer happens to find, and every later comparison would inherit
  that bias.
* **Strata are chosen to over-represent the hard cases** Audit 0063 measured - empty
  candidate sets, fuzzy-only matches, low ranking margins, vague top-1 codes, saturated
  RxNorm sets, exact score ties. A uniform sample would spend most of its 300 records on
  cases that already work.
* **Candidate display order is deterministically shuffled** while `linker_rank` is kept in
  the record. A reviewer shown the linker's ranking tends to ratify it; the benchmark
  would then measure agreement with the system it is supposed to judge.
* **Silver labels are written to a separate directory** and never enter the human pack.

Output (all under an ignored private path)::

    data/private_linker_gold/0065/
        pack.jsonl          300 UNLABELLED records with full candidate evidence
        manifest.json       counts, hashes, seeds, strata
        silver/silver.jsonl high-certainty unique-exact matches, SEPARATE from gold
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.annotation.linker_gold import (  # noqa: E402
    LABEL_DIAGNOSIS,
    LABEL_MEDICATION,
    ONTOLOGY_ICD10,
    ONTOLOGY_RXNORM,
    STATUS_SILVER_UNIQUE_EXACT,
    blank_record,
    classify_icd_specificity,
    document_hash,
    utc_now,
    validate_record,
)
from mednorm_vi.inference.config import PipelineConfig  # noqa: E402
from mednorm_vi.kb.indexing.normalization import normalize_text  # noqa: E402
from mednorm_vi.kb.indexing.retrieval import load_index  # noqa: E402
from mednorm_vi.linking.icd10 import link_icd10  # noqa: E402
from mednorm_vi.linking.rxnorm import link_rxnorm  # noqa: E402
from mednorm_vi.resolution.models import (  # noqa: E402
    BoundaryEvidence,
    EntityHypothesis,
    TypeEvidence,
)
from mednorm_vi.training.governed_splits import resolve_governed_splits  # noqa: E402
from mednorm_vi.training.s1_mention_smoke import iter_jsonl  # noqa: E402

PACK_ID = "linker-gold-0065"
LINKER_VERSION = "competition-v3-l5bc"
CONTEXT_CHARS = 160
MAX_SHOWN = 10

TARGET_DIAGNOSIS = 200
TARGET_MEDICATION = 100

#: Every stratum a mention matches is recorded as a facet; the PRIMARY stratum used for
#: quota allocation is the rarest facet it matches. First-match assignment was tried and
#: rejected: `rx_exact_ranking_tie` and `rx_saturated_20_candidates` apply to almost every
#: medication mention (Audit 0063 §7.4 measured 46.2% ties and 83.3% saturated), so they
#: absorbed the entire population and the strength / dose-form / ingredient-only strata the
#: brief asks for came out empty. Rarest-first guarantees every declared stratum is
#: represented whenever the corpus contains it at all.
DIAGNOSIS_STRATA = (
    "dx_empty_candidate_set",
    "dx_vague_top1_category_or_unspecified",
    "dx_low_margin_ranking",
    "dx_hierarchy_parent_child_ambiguity",
    "dx_exact_kb_alias",
    "dx_len_8plus_words",
    "dx_len_1_word",
    "dx_len_4_7_words",
    "dx_len_2_3_words",
    "dx_fuzzy_only",
)
MEDICATION_STRATA = (
    "rx_exact_ranking_tie",
    "rx_saturated_20_candidates",
    "rx_exact_kb_alias",
    "rx_ingredient_strength_form",
    "rx_strength_bearing",
    "rx_dose_form",
    "rx_brand_like",
    "rx_tty_ambiguity_in_pin_scd_sbd",
    "rx_ingredient_only",
)

STRENGTH_UNITS = ("mg", "g", "ml", "mcg", "µg", "iu", "ui", "%")
DOSE_FORMS = (
    "viên",
    "vien",
    "nang",
    "gói",
    "goi",
    "ống",
    "ong",
    "chai",
    "lọ",
    "lo",
    "tablet",
    "capsule",
    "cream",
    "gel",
    "syrup",
    "siro",
    "dịch",
    "dich",
    "tiêm",
    "tiem",
    "truyền",
    "truyen",
    "kem",
    "thuốc mỡ",
)


def hypothesis_for(index: int, start: int, end: int, text: str, label: str) -> EntityHypothesis:
    return EntityHypothesis(
        hypothesis_id=f"gold-{index:05d}",
        document_id="gold",
        start=start,
        end=end,
        text=text,
        entity_type=label,
        status="accepted",
        chosen_proposal_id="gold",
        source_proposal_ids=("gold",),
        boundary_evidence=BoundaryEvidence(
            policy="gold",
            chosen_kind="gold",
            chosen_proposal_id="gold",
            considered_kinds=("gold",),
        ),
        type_evidence=TypeEvidence(
            entity_type=label, source_specialist="gold", proposed_types=(label,)
        ),
        score=1.0,
    )


def strip_accents(value: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", value) if unicodedata.category(c) != "Mn"
    )


def has_strength(surface: str) -> bool:
    low = strip_accents(surface).lower()
    return any(unit in low for unit in STRENGTH_UNITS) and any(ch.isdigit() for ch in low)


def has_dose_form(surface: str) -> bool:
    low = surface.lower()
    return any(form in low for form in DOSE_FORMS)


def looks_brand_like(surface: str) -> bool:
    """A capitalised single token that is not a strength or a dose form."""
    tokens = surface.split()
    return (
        len(tokens) == 1
        and tokens[0][:1].isupper()
        and not has_strength(surface)
        and not has_dose_form(surface)
    )


def icd_structure(record: dict[str, Any], code: str, index: Any) -> dict[str, Any]:
    metadata = record.get("metadata", {}) if record else {}
    parent = code[:-1] if len(code) > 3 else None
    if parent and parent not in index.records:
        parent = code[:3] if len(code) > 3 else None
    children = sorted(
        other for other in index.graph.get(code, []) if other.startswith(code) and other != code
    )[:8]
    return {
        "ontology": ONTOLOGY_ICD10,
        "parent_code": parent if parent and parent in index.records else None,
        "child_codes": children,
        "hierarchy_depth": max(0, len(code) - 3),
        "specificity_tier": str(metadata.get("specificity", "")) or None,
        "specificity_class": classify_icd_specificity(code),
    }


def rxnorm_structure(record: dict[str, Any], evidence: tuple[str, ...]) -> dict[str, Any]:
    metadata = record.get("metadata", {}) if record else {}
    matched = [e.split(":", 1)[1] for e in evidence if e.startswith("matched:")]
    graph = [e.split(":", 1)[1] for e in evidence if e.startswith("graph:")]
    name = str(record.get("canonical_name", "")) if record else ""
    return {
        "ontology": ONTOLOGY_RXNORM,
        "tty": str(metadata.get("tty", "")) or None,
        "ingredient": name.split(" ")[0] if name else None,
        "precise_ingredient": None,
        "strength": next((t for t in name.split() if any(c.isdigit() for c in t)), None),
        "dose_form": next((f for f in DOSE_FORMS if f in name.lower()), None),
        "brand": name if str(metadata.get("tty", "")) in ("SBD", "BN") else None,
        "graph_relations": graph,
        "structured_match": matched,
    }


def build_candidates(result: Any, index: Any, ontology: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank, candidate in enumerate(result.candidates[:MAX_SHOWN], start=1):
        code = str(candidate.code)
        record = index.records.get(code, {})
        entry: dict[str, Any] = {
            "code": code,
            "canonical_name": str(record.get("canonical_name", "")),
            "aliases": [str(a) for a in (record.get("aliases") or [])][:12],
            "linker_rank": rank,
            "score": float(candidate.score),
            "retrieval_channels": [
                c
                for c in candidate.channels
                if c in ("exact", "exact_ascii", "char_ngram", "sparse", "graph")
            ],
        }
        if ontology == ONTOLOGY_ICD10:
            entry["display_code"] = str(record.get("metadata", {}).get("dotted_code", code))
            entry["structure"] = icd_structure(record, code, index)
        else:
            entry["structure"] = rxnorm_structure(record, tuple(candidate.evidence))
        out.append(entry)
    return out


def diagnosis_facets(
    surface: str, candidates: list[dict[str, Any]], exact_alias: bool
) -> list[str]:
    """Every diagnosis stratum this mention belongs to."""
    facets: list[str] = []
    words = len(surface.split())
    if not candidates:
        facets.append("dx_empty_candidate_set")
    else:
        top = candidates[0]
        if top["structure"]["specificity_class"] in ("category", "unspecified", "other"):
            facets.append("dx_vague_top1_category_or_unspecified")
        if len(candidates) > 1 and abs(top["score"] - candidates[1]["score"]) <= 0.01:
            facets.append("dx_low_margin_ranking")
        codes = {c["code"] for c in candidates}
        if any(c["structure"].get("parent_code") in codes for c in candidates):
            facets.append("dx_hierarchy_parent_child_ambiguity")
    if exact_alias:
        facets.append("dx_exact_kb_alias")
    elif candidates:
        facets.append("dx_fuzzy_only")
    if words == 1:
        facets.append("dx_len_1_word")
    elif words <= 3:
        facets.append("dx_len_2_3_words")
    elif words <= 7:
        facets.append("dx_len_4_7_words")
    else:
        facets.append("dx_len_8plus_words")
    return facets


def medication_facets(
    surface: str, candidates: list[dict[str, Any]], exact_alias: bool, offered_total: int
) -> list[str]:
    """Every medication stratum this mention belongs to."""
    facets: list[str] = []
    if len(candidates) > 1 and abs(candidates[0]["score"] - candidates[1]["score"]) <= 1e-9:
        facets.append("rx_exact_ranking_tie")
    if offered_total >= 20:
        facets.append("rx_saturated_20_candidates")
    if exact_alias:
        facets.append("rx_exact_kb_alias")
    strength, form = has_strength(surface), has_dose_form(surface)
    if strength and form:
        facets.append("rx_ingredient_strength_form")
    if strength:
        facets.append("rx_strength_bearing")
    if form:
        facets.append("rx_dose_form")
    if looks_brand_like(surface):
        facets.append("rx_brand_like")
    ttys = {c["structure"].get("tty") for c in candidates}
    if len(ttys & {"IN", "PIN", "SCD", "SBD", "MIN"}) >= 2:
        facets.append("rx_tty_ambiguity_in_pin_scd_sbd")
    if not strength and not form:
        facets.append("rx_ingredient_only")
    return facets


def primary_stratum(facets: list[str], rarity: dict[str, int], order: tuple[str, ...]) -> str:
    """The rarest facet this mention matches; declaration order breaks ties."""
    if not facets:
        return order[-1]
    return min(facets, key=lambda name: (rarity.get(name, 0), order.index(name)))


def allocate(strata: tuple[str, ...], pools: dict[str, list], target: int) -> dict[str, int]:
    """Even allocation across non-empty strata, redistributing what small strata cannot fill."""
    quotas: dict[str, int] = {name: 0 for name in strata}
    remaining = target
    available = {name: len(pools.get(name, [])) for name in strata}
    active = [name for name in strata if available[name] > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for name in list(active):
            if remaining <= 0:
                break
            take = min(share, available[name] - quotas[name], remaining)
            if take <= 0:
                active.remove(name)
                continue
            quotas[name] += take
            remaining -= take
            progressed = True
        if not progressed:
            break
    return quotas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "private_linker_gold" / "0065")
    parser.add_argument("--diagnosis", type=int, default=TARGET_DIAGNOSIS)
    parser.add_argument("--medication", type=int, default=TARGET_MEDICATION)
    args = parser.parse_args()

    config = PipelineConfig.load(str(REPO / "configs" / "pipeline" / "full_v1.yaml"))
    icd_index = load_index(config.icd_index)
    rx_index = load_index(config.rxnorm_index)
    snapshot = icd_index.source_snapshot_id or "unknown"
    print(f"ICD {len(icd_index.records):,} records | RxNorm {len(rx_index.records):,} records")
    print(f"snapshot {snapshot}")

    corpus = REPO / "data" / "derived" / "training_corpora" / "mednorm_vi_training_v1"
    splits = resolve_governed_splits([corpus], splits=("validation",))
    examples = list(iter_jsonl(splits["validation"].path))
    print(f"validation {splits['validation'].sha256} ({len(examples)} examples)")

    icd_aliases, rx_aliases = set(icd_index.exact), set(rx_index.exact)

    collected: list[dict[str, Any]] = []
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    duplicates = 0
    counter = 0
    silver: list[dict[str, Any]] = []

    for example in examples:
        text = str(example["text"])
        doc_hash = document_hash(text)
        source = str(example.get("source_dataset", ""))
        for entity in example.get("entities", []):
            label = {"DIAGNOSIS": LABEL_DIAGNOSIS, "MEDICATION": LABEL_MEDICATION}.get(
                str(entity["target_type"])
            )
            if label is None:
                continue
            counter += 1
            start, end = int(entity["start"]), int(entity["end"])
            surface = text[start:end]
            normalized = normalize_text(surface)

            # Deduplicate by normalized mention + normalized local context, so the same
            # phrase in the same setting is annotated once.
            key = hashlib.sha256(
                (
                    normalized
                    + "|"
                    + normalize_text(text[max(0, start - 40) : start])
                    + "|"
                    + normalize_text(text[end : end + 40])
                ).encode("utf-8")
            ).hexdigest()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)

            hypothesis = hypothesis_for(counter, start, end, surface, label)
            if label == LABEL_DIAGNOSIS:
                result = link_icd10(hypothesis, icd_index)
                index, ontology, aliases = icd_index, ONTOLOGY_ICD10, icd_aliases
            else:
                result = link_rxnorm(hypothesis, rx_index)
                index, ontology, aliases = rx_index, ONTOLOGY_RXNORM, rx_aliases

            offered_total = len(result.candidates)
            candidates = build_candidates(result, index, ontology)
            exact_alias = normalized in aliases
            facets = (
                diagnosis_facets(surface, candidates, exact_alias)
                if label == LABEL_DIAGNOSIS
                else medication_facets(surface, candidates, exact_alias, offered_total)
            )

            record = blank_record(
                document_text_hash=doc_hash,
                source_dataset=source,
                start=start,
                end=end,
                mention_text=surface,
                context_before=text[max(0, start - CONTEXT_CHARS) : start],
                context_after=text[end : end + CONTEXT_CHARS],
                entity_type=label,
                candidate_snapshot_id=snapshot,
                offered_candidates=candidates,
                provenance={
                    "sampling_seed": args.seed,
                    "stratum": "",  # assigned in the second pass, once rarity is known
                    "facets": facets,
                    "pack_id": PACK_ID,
                    "linker_version": LINKER_VERSION,
                    "validation_split_sha256": splits["validation"].sha256,
                    "offered_total_before_cap": offered_total,
                    "exact_alias_present": exact_alias,
                    "surface_words": len(surface.split()),
                },
            )
            collected.append(record)

            # §7 silver: ONE unique active concept reached by an exact alias. Never mixed
            # into the human pack, never derived from a fuzzy top-1.
            if exact_alias:
                exact_ids = list(dict.fromkeys(index.exact.get(normalized, [])))
                if len(exact_ids) == 1:
                    only = exact_ids[0]
                    metadata = index.records.get(only, {}).get("metadata", {})
                    unambiguous = (
                        ontology == ONTOLOGY_ICD10
                        or len(str(metadata.get("tty_values", "")).split("|")) == 1
                    )
                    if unambiguous and str(metadata.get("membership", "")) == "searchable":
                        silver_record = json.loads(json.dumps(record))
                        silver_record["selected_codes"] = [only]
                        silver_record["adjudication_status"] = STATUS_SILVER_UNIQUE_EXACT
                        silver_record["reviewer_id"] = "auto:unique_exact_alias"
                        silver_record["reviewer_confidence"] = None
                        silver_record["adjudication_notes"] = (
                            "Automatic: the normalized mention matches exactly one active "
                            "searchable concept by exact alias. NOT human gold."
                        )
                        silver_record["updated_at"] = utc_now()
                        silver.append(silver_record)

    # Second pass: rarest-facet assignment. Rarity is global over the whole eligible
    # population, so a stratum that barely occurs claims its mentions before an abundant
    # one can swallow them.
    rarity: Counter[str] = Counter()
    for record in collected:
        rarity.update(record["provenance"]["facets"])
    for record in collected:
        order = DIAGNOSIS_STRATA if record["entity_type"] == LABEL_DIAGNOSIS else MEDICATION_STRATA
        stratum = primary_stratum(record["provenance"]["facets"], rarity, order)
        record["provenance"]["stratum"] = stratum
        pools[stratum].append(record)

    print(f"\neligible gold mentions {counter} | deduplicated away {duplicates}")
    print(f"distinct mentions pooled {sum(len(v) for v in pools.values())}")
    print("facet frequency over the eligible population:")
    for name, count in sorted(rarity.items()):
        print(f"  {name:<42}{count:>6}")

    rng = random.Random(args.seed)
    chosen: list[dict[str, Any]] = []
    plan = ((DIAGNOSIS_STRATA, args.diagnosis), (MEDICATION_STRATA, args.medication))
    for strata, target in plan:
        quotas = allocate(strata, pools, target)
        for name in strata:
            pool = sorted(pools.get(name, []), key=lambda r: r["annotation_id"])
            take = min(quotas[name], len(pool))
            if take:
                chosen.extend(rng.sample(pool, take))

    # Deterministic display shuffle; linker_rank survives inside each candidate.
    for record in chosen:
        order_seed = int(record["annotation_id"][4:12], 16) ^ args.seed
        shuffler = random.Random(order_seed)
        shuffler.shuffle(record["offered_candidates"])
        record["provenance"]["display_order_seed"] = order_seed
    chosen.sort(key=lambda r: r["annotation_id"])

    problems = 0
    for record in chosen:
        outcome = validate_record(record)
        if not outcome.valid:
            problems += 1
            print(f"  INVALID {record['annotation_id']}: {outcome.problems[:2]}")
    if problems:
        raise SystemExit(f"{problems} record(s) failed validation; refusing to write the pack")

    out = args.out
    (out / "silver").mkdir(parents=True, exist_ok=True)
    pack_path = out / "pack.jsonl"
    pack_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in chosen),
        encoding="utf-8",
    )
    silver_path = out / "silver" / "silver.jsonl"
    silver_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in silver),
        encoding="utf-8",
    )

    by_stratum = Counter(r["provenance"]["stratum"] for r in chosen)
    by_type = Counter(r["entity_type"] for r in chosen)
    by_source = Counter(r["source_dataset"] for r in chosen)
    manifest = {
        "pack_id": PACK_ID,
        "created_at": utc_now(),
        "sampling_seed": args.seed,
        "linker_version": LINKER_VERSION,
        "candidate_snapshot_id": snapshot,
        "validation_split_sha256": splits["validation"].sha256,
        "icd_index": config.icd_index,
        "rxnorm_index": config.rxnorm_index,
        "eligible_gold_mentions": counter,
        "deduplicated_away": duplicates,
        "records": len(chosen),
        "records_by_entity_type": dict(by_type),
        "records_by_source": dict(by_source),
        "records_by_stratum": dict(sorted(by_stratum.items())),
        "diagnosis_strata_available": {k: len(pools.get(k, [])) for k in DIAGNOSIS_STRATA},
        "medication_strata_available": {k: len(pools.get(k, [])) for k in MEDICATION_STRATA},
        "facet_coverage_in_pack": {
            name: sum(1 for r in chosen if name in r["provenance"]["facets"])
            for name in sorted(set(DIAGNOSIS_STRATA) | set(MEDICATION_STRATA))
        },
        "facet_frequency_in_population": dict(sorted(rarity.items())),
        "silver_records": len(silver),
        "pack_sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest(),
        "silver_sha256": hashlib.sha256(silver_path.read_bytes()).hexdigest(),
        "contains_clinical_text": True,
        "storage_policy": (
            "PRIVATE. This pack contains bounded windows of clinical text and must never be "
            "tracked in Git. data/** is gitignored."
        ),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    print(f"\npack     {pack_path}  ({len(chosen)} records)  sha256 {manifest['pack_sha256']}")
    print(f"silver   {silver_path}  ({len(silver)} records)  sha256 {manifest['silver_sha256']}")
    print(f"manifest {manifest_path}  sha256 {manifest_sha}")
    print(f"\nby entity type: {dict(by_type)}")
    print("by stratum:")
    for name, count in sorted(by_stratum.items()):
        print(f"  {name:<42}{count:>5}   (pool {len(pools.get(name, []))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
