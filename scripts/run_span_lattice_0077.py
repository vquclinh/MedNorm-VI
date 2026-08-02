#!/usr/bin/env python3
"""Experiment 0077: finite span-lattice boundary corrector (E3 + Qwen3-8B).

    note + E3 entities -> deterministic same-type clusters -> finite span options (code)
                       -> ONE Qwen3-8B generation per document, ids only
                       -> KEEP_ORIGINALS unless a listed candidate is chosen
                       -> candidates = [] for every CHẨN_ĐOÁN / THUỐC

Type is FROZEN from E3 - the 0076 smoke showed semantic retyping is unreliable. Every emitted
span is a literal slice of the source produced before the model ran, so arbitrary text,
offsets, types and codes are all unrepresentable.

Deployed: E3 135,002,117 + Qwen3-8B 8,190,735,360 = 8,325,737,477 (< 9B).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.reasoner import budget  # noqa: E402
from mednorm_vi.reasoner.lattice import (  # noqa: E402
    LATTICE_VERSION,
    apply_selection,
    build_clusters,
)
from mednorm_vi.reasoner.prompt import SYSTEM, lattice_prompt  # noqa: E402
from mednorm_vi.reasoner.validator import ASSERTION_TYPES, CANDIDATE_TYPES  # noqa: E402

_JSON = re.compile(r"\{.*\}", re.S)


def log(message: str) -> None:
    print(f"[0077] {message}", flush=True)


def parse_clusters(text: str) -> dict[int, list[str]] | None:
    match = _JSON.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    rows = payload.get("clusters") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    out: dict[int, list[str]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), int):
            selected = row.get("select")
            if isinstance(selected, list):
                out[row["cluster_id"]] = [str(s) for s in selected]
    return out


def inherit_assertions(
    entities: list[dict[str, Any]], proposals: tuple[dict[str, Any], ...]
) -> None:
    """Conservative deterministic inheritance for replaced spans.

    A merged span inherits a flag only when EVERY original proposal it replaces carried it.
    Unanimity is the conservative direction: a flag set on one fragment is weak evidence about
    the whole phrase, and a false negation is a clinical error.
    """
    sets = [set(p.get("assertions") or []) for p in proposals if p.get("assertions") is not None]
    shared = set.intersection(*sets) if sets else set()
    for entity in entities:
        if entity["type"] in ASSERTION_TYPES:
            entity["assertions"] = sorted(shared)
        if entity["type"] in CANDIDATE_TYPES:
            entity["candidates"] = []


class LatticeModel:
    def __init__(self, model_root: Path, device: str, max_new_tokens: int) -> None:
        self.model_root, self.device = model_root, device
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = str(self.model_root)
        self._tokenizer = AutoTokenizer.from_pretrained(source)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._model = (
            AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype).to(self.device).eval()
        )
        config = self._model.generation_config
        config.do_sample = False
        for name in ("temperature", "top_p", "top_k"):
            if hasattr(config, name):
                setattr(config, name, None)
        log("Qwen3-8B loaded (greedy, no sampling parameters)")

    def ask(self, user: str) -> dict[int, list[str]] | None:
        self._load()
        import torch

        text = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self._tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return parse_clusters(
            self._tokenizer.decode(
                generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )
        )


def load_note(path: Path) -> str:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("text", "content", "note"):
                if key in payload:
                    return str(payload[key])
        return str(payload)
    return path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--seed-entities", type=Path, required=True)
    parser.add_argument("--seed-run-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--expected-documents", type=int, default=100)
    parser.add_argument("--limit-documents", type=int, default=0)
    parser.add_argument("--documents", default="", help="comma-separated stems to run, e.g. 1,10")
    args = parser.parse_args(argv)

    if "semantic_s1_0072" in (
        args.seed_run_config.read_text(encoding="utf-8") if args.seed_run_config.is_file() else ""
    ):
        print(
            "BLOCKED: seed uses the Embedding-4B/Reranker-4B pair; with Qwen3-8B that is "
            "16,369,296,389 parameters against a 9,000,000,000 cap.",
            file=sys.stderr,
        )
        return 2
    total = budget.assert_within_cap(with_semantic_s1=False)
    log(f"seed OK (E3-only) | deployed parameters {total:,}")

    notes = sorted(p for p in args.input_dir.glob("*") if p.is_file())
    if args.documents:
        wanted = {s.strip() for s in args.documents.split(",") if s.strip()}
        notes = [p for p in notes if p.stem in wanted]
    smoke = bool(args.limit_documents or args.documents)
    if args.limit_documents:
        notes = notes[: args.limit_documents]
    if smoke:
        log(f"SMOKE: {len(notes)} documents ({[p.stem for p in notes]}) - NOT a submission")
    if not notes:
        print("BLOCKED: no documents selected", file=sys.stderr)
        return 2

    model = LatticeModel(args.model_root, args.device, args.max_new_tokens)
    out_dir = args.run_dir / "output_raw_dotless"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    stats: Counter[str] = Counter()
    per_type_before: Counter[str] = Counter()
    per_type_after: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for position, path in enumerate(notes, start=1):
        note = load_note(path)
        seed_path = args.seed_entities / f"{path.stem}.json"
        proposals = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.is_file() else []
        stats["e3_entities_in"] += len(proposals)
        for entity in proposals:
            per_type_before[entity.get("type", "")] += 1

        clusters = build_clusters(note, proposals)
        stats["clusters_total"] += len(clusters)
        decidable = [c for c in clusters if len(c.candidates) > 1]
        stats["clusters_with_multiple_proposals"] += sum(1 for c in clusters if c.multiple)
        stats["lattice_candidates_total"] += sum(len(c.candidates) for c in clusters)

        selection: dict[int, list[str]] = {}
        if decidable:
            reply = model.ask(lattice_prompt(note, decidable))
            if reply is None:
                stats["parse_fallbacks"] += 1
            else:
                selection = reply

        entities: list[dict[str, Any]] = []
        for cluster in clusters:
            chosen = selection.get(cluster.cluster_id)
            produced = apply_selection(cluster, chosen, note)
            changed = [(e["text"], tuple(e["position"])) for e in produced] != [
                (p["text"], tuple(p["position"])) for p in cluster.proposals
            ]
            if changed:
                stats["clusters_changed"] += 1
                if len(produced) < len(cluster.proposals):
                    stats["fragments_removed"] += len(cluster.proposals) - len(produced)
                    stats["merged_entities_created"] += len(produced)
                else:
                    stats["boundary_expansions"] += 1
                if len(examples) < 25:
                    start = max(0, cluster.proposals[0]["position"][0] - 40)
                    examples.append(
                        {
                            "document": path.stem,
                            "context": note[start : start + 160],
                            "original": [
                                {"text": p["text"], "position": p["position"], "type": p["type"]}
                                for p in cluster.proposals
                            ],
                            "selected": [
                                {"text": e["text"], "position": e["position"], "type": e["type"]}
                                for e in produced
                            ],
                            "provenance": [
                                c.provenance
                                for c in cluster.candidates
                                if c.candidate_id in (chosen or [])
                            ],
                        }
                    )
            else:
                stats["clusters_keep_original"] += 1
            inherit_assertions(produced, cluster.proposals)
            entities.extend(produced)

        entities.sort(key=lambda e: (e["position"][0], e["position"][1], e["type"]))
        for entity in entities:
            per_type_after[entity["type"]] += 1
        stats["entities_out"] += len(entities)
        (out_dir / f"{path.stem}.json").write_text(
            json.dumps(entities, ensure_ascii=False), encoding="utf-8"
        )
        if position % 5 == 0 or position == len(notes):
            log(f"  {position}/{len(notes)} docs")

    # Diagnostics are computed from the SERIALIZED output, not from raw model decisions -
    # the 0076 smoke reported 9 type changes while the files showed 7.
    report = {
        "lattice_version": LATTICE_VERSION,
        "documents": len(notes),
        **{
            k: stats[k]
            for k in (
                "e3_entities_in",
                "entities_out",
                "clusters_total",
                "clusters_with_multiple_proposals",
                "lattice_candidates_total",
                "clusters_changed",
                "clusters_keep_original",
                "fragments_removed",
                "merged_entities_created",
                "boundary_expansions",
                "parse_fallbacks",
            )
        },
        "per_type_before": dict(sorted(per_type_before.items())),
        "per_type_after": dict(sorted(per_type_after.items())),
        "type_changes": 0,
        "type_frozen": True,
        "candidate_policy": "forced_all_null",
        "assertion_inheritance": "unanimous_across_replaced_proposals",
        "changed_cluster_examples": examples,
        "model": budget.QWEN3_8B,
        "revision": budget.QWEN3_8B_REVISION,
        "deployed_parameters": total,
        "training_performed": False,
        "submitted": False,
        "runtime_seconds": round(time.time() - started, 1),
        "contains_clinical_text": bool(examples),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "diagnostic-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    log(
        f"in {stats['e3_entities_in']} -> out {stats['entities_out']} | "
        f"clusters changed {stats['clusters_changed']}/{stats['clusters_total']} | "
        f"fragments removed {stats['fragments_removed']} | "
        f"merged {stats['merged_entities_created']} | "
        f"expansions {stats['boundary_expansions']} | "
        f"fallbacks {stats['parse_fallbacks']}"
    )
    for example in examples[:8]:
        log(
            f"  [{example['document']}] {[o['text'] for o in example['original']]} -> "
            f"{[s['text'] for s in example['selected']]}"
        )
    if smoke:
        log("smoke complete - inspect diagnostic-summary.json")
        return 0
    if len(sorted(out_dir.glob("*.json"))) != args.expected_documents:
        print("BLOCKED: document count mismatch", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
