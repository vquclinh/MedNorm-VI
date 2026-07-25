"""Deterministic governed training-corpus builder (Audit 0017).

Runs the existing public-NER adapters over the governance-approved datasets,
applies versioned v1 label mappings (only concrete MAP_EXACT / MAP_APPROXIMATE
targets that are allowed for type training), validates half-open offsets, groups
examples for leakage control, produces group-based train / validation /
internal_test splits, and writes canonical JSONL + manifests + quality reports.

Local, deterministic, offline. No neural models. Emits aggregate reports only —
never raw clinical text into tracked files (all outputs go under an ignored
derived namespace).
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..resources.ner import load_ner_manifest
from ..schemas.constants import ENTITY_TYPES, TYPE_BY_ORGANIZER_LABEL
from .public_ner import (
    bio_examples_to_documents,
    load_json_records,
    read_conll,
    require_training_allowed,
    vimq_records_to_documents,
)

BUILDER_VERSION = "governed-corpus-v1"
SCHEMA_VERSION = "canonical_example_v1"
DEFAULT_SEED = 20260723

# Split fractions for internal grouped split (train / validation / internal_test).
_VAL_FRAC = 0.15
_TEST_FRAC = 0.15


@dataclass(frozen=True, slots=True)
class SourceSpec:
    dataset_id: str
    manifest: str
    mapping: str
    # (split_name, file_path, reader) tuples
    conll_splits: tuple[tuple[str, str], ...] = ()
    vimq_splits: tuple[tuple[str, str], ...] = ()
    source_revision: str = ""


SPLIT_POLICY_VERSION = "split-policy-v2-duplicate-family"


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", " ".join(text.split())).casefold()


def _group_id(text: str) -> str:
    return "g_" + hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()[:16]


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _normalized_key(text: str) -> str:
    """Merges case/accent/punctuation/whitespace-only variants deterministically."""
    t = _strip_accents(unicodedata.normalize("NFC", text)).casefold()
    t = "".join(c if (c.isalnum() or c.isspace()) else " " for c in t)
    return " ".join(t.split())


def _simhash(text: str, bits: int = 64) -> int:
    """Deterministic SimHash over character 4-grams (no neural embeddings)."""
    key = _normalized_key(text)
    grams = [key[i:i + 4] for i in range(max(1, len(key) - 3))] or [key]
    v = [0] * bits
    for g in grams:
        h = int(hashlib.sha1(g.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= (1 << i)
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


_SIMHASH_BANDS = 4  # four 16-bit bands over a 64-bit SimHash (LSH for Hamming <= 3)


def simhash_bands(s: int) -> tuple[int, ...]:
    """The four 16-bit bands (bits 0-15, 16-31, 32-47, 48-63) of a 64-bit SimHash."""
    return tuple((s >> (16 * b)) & 0xFFFF for b in range(_SIMHASH_BANDS))


def near_duplicate_pairs(hashes: list[int], *, threshold: int = 3) -> list[tuple[int, int]]:
    """Complete-recall near-duplicate pairs for 64-bit SimHash Hamming <= threshold.

    Uses four-band LSH candidate generation (a pair with Hamming <= 3 differs in <= 3
    bits, touching <= 3 of the 4 bands, so it shares at least one identical band),
    then verifies each unique candidate by the full 64-bit Hamming distance. Returns
    sorted (i, j) index pairs (i < j). Deterministic.
    """
    band_buckets: dict[tuple[int, int], list[int]] = {}
    for i, s in enumerate(hashes):
        for b, bv in enumerate(simhash_bands(s)):
            band_buckets.setdefault((b, bv), []).append(i)
    checked: set[tuple[int, int]] = set()
    accepted: list[tuple[int, int]] = []
    for members in band_buckets.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                x, y = members[a], members[b]
                pair = (x, y) if x < y else (y, x)
                if pair in checked:
                    continue
                checked.add(pair)
                if _hamming(hashes[x], hashes[y]) <= threshold:
                    accepted.append(pair)
    return sorted(accepted)


def duplicate_families(texts: list[str], *, near_dup_hamming: int = 3) -> dict[str, object]:
    """Deterministic duplicate-family clustering over example texts.

    Unions three family kinds:
      * exact-byte duplicates;
      * normalized duplicates (case/accent/punctuation/whitespace folded);
      * near-duplicates via 64-bit SimHash with a **four-band LSH** candidate step
        (bands = bits 0-15/16-31/32-47/48-63). Any pair with Hamming distance <=
        `near_dup_hamming` (<= 3) differs in <= 3 bit positions, which touch at most
        3 of the 4 bands, so the pair shares at least one identical 16-bit band and
        is generated as a candidate (complete recall for Hamming <= 3). Candidates
        are then verified by the full 64-bit Hamming distance.

    Returns stable family IDs (hashes only — no raw text) plus aggregate group stats.
    """
    n = len(texts)
    uf = _UnionFind(n)
    exact: dict[str, int] = {}
    norm: dict[str, int] = {}
    exact_groups = 0
    for i, t in enumerate(texts):
        eb = hashlib.sha256(t.encode("utf-8")).hexdigest()
        if eb in exact:
            uf.union(i, exact[eb])
        else:
            exact[eb] = i
            exact_groups += 1
        nk = _normalized_key(t)
        if nk in norm:
            uf.union(i, norm[nk])
        else:
            norm[nk] = i
    # near-duplicate: four-band LSH candidate generation + full-Hamming verification
    sims = [_simhash(t) for t in texts]
    pairs = near_duplicate_pairs(sims, threshold=near_dup_hamming)
    for x, y in pairs:
        uf.union(x, y)
    near_pairs = len(pairs)
    # stable family id from the minimum normalized key in each component
    comp_examples: dict[int, list[int]] = {}
    for i in range(n):
        comp_examples.setdefault(uf.find(i), []).append(i)
    family_id: list[str] = [""] * n
    sizes: dict[int, int] = {}
    for members in comp_examples.values():
        rep = min(_normalized_key(texts[m]) for m in members)
        fid = "fam_" + hashlib.sha256(rep.encode("utf-8")).hexdigest()[:16]
        for m in members:
            family_id[m] = fid
        sizes[len(members)] = sizes.get(len(members), 0) + 1
    return {
        "family_id": family_id,
        "exact_groups": exact_groups,
        "normalized_groups": len(norm),
        "near_duplicate_candidate_pairs": near_pairs,
        "families": len(comp_examples),
        "family_size_distribution": dict(sorted(sizes.items())),
    }


def _internal_type(target: str) -> str:
    """Translate a Vietnamese organizer label to the internal English enum.
    Accepts an internal enum unchanged; returns '' for non-type targets."""
    if target in ENTITY_TYPES:
        return target
    return TYPE_BY_ORGANIZER_LABEL.get(target, "")


def concrete_type_mapping(mapping_path: str | Path) -> dict[str, str]:
    """source_label -> internal ENTITY_TYPE for labels allowed for type training."""
    doc = yaml.safe_load(Path(mapping_path).read_text(encoding="utf-8")) or {}
    out: dict[str, str] = {}
    for m in doc.get("mappings", []) or []:
        internal = _internal_type(str(m.get("target", "")))
        cls = str(m.get("classification", ""))
        if (internal in ENTITY_TYPES
                and cls in {"MAP_EXACT", "MAP_APPROXIMATE"}
                and bool(m.get("allowed_for_type_training", False))):
            out[str(m["source_label"])] = internal
    return out


def _mapping_status_by_target(mapping_path: str | Path) -> dict[str, dict[str, Any]]:
    """Keyed by internal ENTITY_TYPE (so a.entity_type lookups match)."""
    doc = yaml.safe_load(Path(mapping_path).read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, Any]] = {}
    for m in doc.get("mappings", []) or []:
        internal = _internal_type(str(m.get("target", "")))
        if internal:
            out[internal] = {
                "source_label": str(m.get("source_label", "")),
                "classification": str(m.get("classification", "")),
                "exclude_from_validation": bool(m.get("exclude_from_validation", False)),
            }
    return out


def _default_sources(base: Path) -> list[SourceSpec]:
    pn = base / "phoner_covid19" / "data" / "word"
    vm = base / "vimedner" / "data"
    vq = base / "vimq" / "data"
    md = "data/manifests"
    lm = "configs/resources/label_mappings/public_ner"
    return [
        SourceSpec("phoner_covid19", f"{md}/phoner-covid19-public-ner.yaml",
                   f"{lm}/phoner_covid19_v1.yaml",
                   conll_splits=(("train", str(pn / "train_word.conll")),
                                 ("dev", str(pn / "dev_word.conll")),
                                 ("test", str(pn / "test_word.conll")))),
        SourceSpec("vimedner", f"{md}/vimedner-public-ner.yaml",
                   f"{lm}/vimedner_v1.yaml",
                   conll_splits=(("train", str(vm / "train.txt")),
                                 ("dev", str(vm / "dev.txt")),
                                 ("test", str(vm / "test.txt")))),
        SourceSpec("vimq", f"{md}/vimq-public-ner.yaml",
                   f"{lm}/vimq_v1.yaml",
                   vimq_splits=(("train", str(vq / "train.json")),
                                ("dev", str(vq / "dev.json")),
                                ("test", str(vq / "test.json")))),
    ]


def _example_dict(doc: Any, spec: SourceSpec, split: str,
                  status_by_target: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entities = []
    for a in doc.annotations:
        assert doc.text[a.span.start:a.span.end] == a.text, "offset invariant violated"
        meta = status_by_target.get(a.entity_type, {})
        entities.append({
            "text": a.text, "start": a.span.start, "end": a.span.end,
            "target_type": a.entity_type,
            "source_label": meta.get("source_label", ""),
            "mapping_status": meta.get("classification", ""),
            "mapping_confidence": 0.9 if meta.get("classification") == "MAP_EXACT" else 0.7,
            "provenance": f"{spec.dataset_id}:{split}",
            "quality_flags": [],
        })
    return {
        "example_id": doc.document_id,
        "document_id": doc.document_id,
        "text": doc.text,
        "entities": entities,
        "source_dataset": spec.dataset_id,
        "source_revision": spec.source_revision,
        "source_split": split,
        "source_record_id": doc.document_id.split(":")[-1],
        "provenance": BUILDER_VERSION,
        "quality_flags": [],
        "group_ids": [_group_id(doc.text)],
        "transformation_version": SCHEMA_VERSION,
    }


def build_governed_corpus(
    base_dir: str | Path = "data/external/public_ner",
    out_dir: str | Path = "data/derived/training_corpora/mednorm_vi_training_v1",
    *, seed: int = DEFAULT_SEED, include_vietmed: bool | None = None,
    vietmed_artifact_dir: str | Path = "data/derived/training_corpora/vietmed_ner_v1",
) -> dict[str, Any]:
    """Build the governed corpus from the text-readable public datasets.

    VietMed-NER inclusion: ``include_vietmed=None`` (auto) includes it only if valid
    preprocessing artifacts have been returned to ``vietmed_artifact_dir``; ``True``
    requires them (fail-fast if absent/corrupt); ``False`` excludes them. There is no
    silent fallback — a present-but-corrupt artifact raises.
    """
    from .vietmed_ner import load_vietmed_artifacts

    base = Path(base_dir)
    out = Path(out_dir)

    # Resolve VietMed inclusion up-front (fail fast; no silent fallback).
    vietmed_examples = None
    if include_vietmed is not False:
        vietmed_examples = load_vietmed_artifacts(vietmed_artifact_dir)  # None if absent
    if include_vietmed is True and vietmed_examples is None:
        raise FileNotFoundError(
            f"include_vietmed=True but no valid VietMed artifacts under {vietmed_artifact_dir}")

    for sub in ("canonical_examples", "splits", "manifests", "quality"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    examples: list[dict[str, Any]] = []
    source_contrib: dict[str, dict[str, Any]] = {}
    coverage: dict[str, Any] = {}
    for spec in _default_sources(base):
        manifest = load_ner_manifest(spec.manifest)
        require_training_allowed(manifest)  # governance gate (fail-fast)
        cmap = concrete_type_mapping(spec.mapping)
        smap = _mapping_status_by_target(spec.mapping)
        n_docs = n_ent = 0
        type_counts: dict[str, int] = {}
        for split, path in spec.conll_splits:
            if not Path(path).is_file():
                continue
            ex = read_conll(path, split=split)
            docs = bio_examples_to_documents(spec.dataset_id, split, ex, cmap)
            for d in docs:
                examples.append(_example_dict(d, spec, split, smap))
                n_docs += 1
                for a in d.annotations:
                    n_ent += 1
                    type_counts[a.entity_type] = type_counts.get(a.entity_type, 0) + 1
        for split, path in spec.vimq_splits:
            if not Path(path).is_file():
                continue
            recs = load_json_records(path).records
            docs = vimq_records_to_documents(spec.dataset_id, split, recs, cmap)
            for d in docs:
                examples.append(_example_dict(d, spec, split, smap))
                n_docs += 1
                for a in d.annotations:
                    n_ent += 1
                    type_counts[a.entity_type] = type_counts.get(a.entity_type, 0) + 1
        source_contrib[spec.dataset_id] = {
            "documents": n_docs, "entities": n_ent, "type_counts": type_counts,
            "concrete_mapping": cmap,
        }
        coverage[spec.dataset_id] = {
            "boundary": bool(cmap), "entity_type": bool(cmap),
            "assertions": manifest.assertion_availability != "none",
            "icd_candidates": manifest.normalization_availability in ("icd", "both"),
            "rxnorm_candidates": manifest.normalization_availability in ("rxnorm", "both"),
            "test_result_pairing": False, "patient_context": False, "relations": False,
        }

    # --- optional VietMed-NER inclusion from returned Colab artifacts (resolved above) ---
    vietmed_status = "not_included_pending"
    if vietmed_examples is not None:
        ent = sum(len(e["entities"]) for e in vietmed_examples)
        examples.extend(vietmed_examples)
        source_contrib["vietmed_ner"] = {
            "documents": len(vietmed_examples), "entities": ent, "type_counts": {},
            "concrete_mapping": {"DRUGCHEMICAL": "MEDICATION"},
        }
        coverage["vietmed_ner"] = {
            "boundary": True, "entity_type": True, "assertions": False,
            "icd_candidates": False, "rxnorm_candidates": False,
            "test_result_pairing": False, "patient_context": False, "relations": False,
        }
        vietmed_status = "included_from_artifacts"

    # --- offset validation (already asserted per-entity; recount) ---
    off_total = off_bad = 0
    for e in examples:
        for ent in e["entities"]:
            off_total += 1
            if e["text"][ent["start"]:ent["end"]] != ent["text"]:
                off_bad += 1

    # --- duplicate-family clustering (exact/normalized/near-duplicate) ---
    fam = duplicate_families([e["text"] for e in examples])
    family_id: list[str] = fam["family_id"]  # type: ignore[assignment]
    for i, e in enumerate(examples):
        e["family_id"] = family_id[i]

    # --- split-quality policy (v2) -----------------------------------------
    # An example is eval-eligible only if it has >=1 entity and EVERY entity is a
    # MAP_EXACT mapping not flagged exclude_from_validation. A family is
    # eval-eligible only if ALL its examples are. Approximate/empty/repaired
    # examples (and any family containing one) stay in train. No family crosses
    # splits.
    def _eval_eligible(ex: dict[str, Any]) -> bool:
        ents = ex["entities"]
        return bool(ents) and all(
            en.get("mapping_status") == "MAP_EXACT" and not en.get("exclude_from_validation")
            for en in ents
        )

    fam_examples: dict[str, list[int]] = {}
    for i, e in enumerate(examples):
        fam_examples.setdefault(e["family_id"], []).append(i)
    fam_eligible = {
        f: all(_eval_eligible(examples[i]) for i in idxs)
        for f, idxs in fam_examples.items()
    }
    eligible_fams = sorted(f for f, ok in fam_eligible.items() if ok)
    # deterministic seed-rotated assignment of eligible families to val/test
    if eligible_fams:
        rot = seed % len(eligible_fams)
        eligible_fams = eligible_fams[rot:] + eligible_fams[:rot]
    n_elig = len(eligible_fams)
    n_val = int(n_elig * _VAL_FRAC)
    n_test = int(n_elig * _TEST_FRAC)
    assign: dict[str, str] = {}
    for idx, f in enumerate(eligible_fams):
        if idx < n_test:
            assign[f] = "internal_test"
        elif idx < n_test + n_val:
            assign[f] = "validation"
        else:
            assign[f] = "train"
    split_examples: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [], "internal_test": []}
    for e in examples:
        split_examples[assign.get(e["family_id"], "train")].append(e)

    # --- write payloads (ignored) ---
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
        h = hashlib.sha256()
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                line = json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                fh.write(line)
                h.update(line.encode("utf-8"))
        return h.hexdigest()

    all_hash = _write_jsonl(out / "canonical_examples" / "public_ner_examples.jsonl", examples)
    split_hashes = {
        s: _write_jsonl(out / "splits" / f"{s}.jsonl", split_examples[s])
        for s in ("train", "validation", "internal_test")
    }

    # leakage cross-split check (by duplicate FAMILY, not exact hash)
    split_fams = {s: {e["family_id"] for e in split_examples[s]} for s in split_examples}
    cross = 0
    keys = list(split_fams)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cross += len(split_fams[keys[i]] & split_fams[keys[j]])

    def _type_dist(rows: list[dict[str, Any]]) -> dict[str, int]:
        d: dict[str, int] = {}
        for e in rows:
            for ent in e["entities"]:
                d[ent["target_type"]] = d.get(ent["target_type"], 0) + 1
        return d

    type_dist = _type_dist(examples)
    per_split_type = {s: _type_dist(split_examples[s]) for s in split_examples}
    per_split_mapping = {
        s: {
            "MAP_EXACT": sum(1 for e in split_examples[s] for en in e["entities"]
                             if en.get("mapping_status") == "MAP_EXACT"),
            "MAP_APPROXIMATE": sum(1 for e in split_examples[s] for en in e["entities"]
                                   if en.get("mapping_status") == "MAP_APPROXIMATE"),
        }
        for s in split_examples
    }
    per_split_source = {
        s: {ds: sum(1 for e in split_examples[s] if e["source_dataset"] == ds)
            for ds in sorted({e["source_dataset"] for e in examples})}
        for s in split_examples
    }

    corpus_manifest = {
        "schema_version": SCHEMA_VERSION, "builder_version": BUILDER_VERSION, "seed": seed,
        "total_examples": len(examples), "total_entities": off_total,
        "examples_jsonl_sha256": all_hash,
        "sources": {k: v["concrete_mapping"] for k, v in source_contrib.items()},
        "target_types_covered": sorted(type_dist),
        "target_types_missing": sorted(set(ENTITY_TYPES) - set(type_dist)),
        "vietmed_status": vietmed_status,
        "notes": ("VietMed-NER included from returned Colab artifacts."
                  if vietmed_status == "included_from_artifacts"
                  else "VietMed-NER not included (pending Colab pyarrow preprocessing)."),
    }
    eval_approx = (per_split_mapping["validation"]["MAP_APPROXIMATE"]
                   + per_split_mapping["internal_test"]["MAP_APPROXIMATE"])
    split_manifest = {
        "seed": seed, "split_policy_version": SPLIT_POLICY_VERSION,
        "fractions": {"validation": _VAL_FRAC, "internal_test": _TEST_FRAC},
        "counts": {s: len(split_examples[s]) for s in split_examples},
        "family_counts": {s: len(split_fams[s]) for s in split_fams},
        "per_split_type": per_split_type,
        "per_split_mapping_status": per_split_mapping,
        "per_split_source": per_split_source,
        "sha256": split_hashes,
        "cross_split_family_leakage": cross,
        "eval_map_approximate_entities": eval_approx,
    }
    leakage = {
        "grouping": "duplicate_family (exact + normalized + simhash near-dup)",
        "exact_groups": fam["exact_groups"],
        "normalized_groups": fam["normalized_groups"],
        "near_duplicate_candidate_pairs": fam["near_duplicate_candidate_pairs"],
        "duplicate_families": fam["families"],
        "family_size_distribution": fam["family_size_distribution"],
        "cross_split_family_leakage_before_policy": None,
        "cross_split_family_leakage_after_policy": cross,
    }

    def _dump(rel: str, payload: Any) -> None:
        (out / rel).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    _dump("manifests/corpus_manifest.json", corpus_manifest)
    _dump("manifests/split_manifest.json", split_manifest)
    _dump("manifests/source_contributions.json", source_contrib)
    _dump("manifests/annotation_coverage.json", coverage)
    _dump("quality/offset_validation.json", {"checked": off_total, "invalid": off_bad})
    _dump("quality/leakage_summary.json", leakage)
    _dump("quality/label_distribution.json", {"by_target_type": type_dist})

    contrib = {
        k: {"documents": v["documents"], "entities": v["entities"]}
        for k, v in source_contrib.items()
    }
    return {
        "total_examples": len(examples), "total_entities": off_total,
        "offset_invalid": off_bad, "type_distribution": type_dist,
        "vietmed_status": vietmed_status,
        "split_counts": split_manifest["counts"], "split_sha256": split_hashes,
        "cross_split_family_leakage": cross,
        "eval_map_approximate_entities": eval_approx,
        "duplicate_families": fam["families"],
        "near_duplicate_candidate_pairs": fam["near_duplicate_candidate_pairs"],
        "per_split_type": per_split_type,
        "target_types_missing": corpus_manifest["target_types_missing"],
        "source_contributions": contrib,
        "out_dir": str(out),
    }


__all__ = [
    "build_governed_corpus", "concrete_type_mapping", "duplicate_families",
    "near_duplicate_pairs", "simhash_bands",
    "BUILDER_VERSION", "SCHEMA_VERSION", "SPLIT_POLICY_VERSION",
]
