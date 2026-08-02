#!/usr/bin/env python3
"""Experiment 0078: strict merge-only span lattice (E3 + Qwen3-8B).

The only change a document may undergo is replacing >=2 same-type E3 fragments with their
exact union. Single proposals are never offered, so 0077's boundary expansions
(`'22' -> '22 tuần'`) are structurally impossible rather than merely discouraged.

Deployed: E3 135,002,117 + Qwen3-8B 8,190,735,360 = 8,325,737,477 (< 9B).
Candidates forced []. Assertions frozen. Type frozen. Nothing trained.
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
from mednorm_vi.reasoner.merge_only import (  # noqa: E402
    MERGE_ONLY_VERSION,
    MERGE_TO_UNION,
    apply_merge,
    build_merge_clusters,
)
from mednorm_vi.reasoner.prompt import SYSTEM, merge_only_prompt  # noqa: E402
from mednorm_vi.reasoner.validator import ASSERTION_TYPES, CANDIDATE_TYPES  # noqa: E402

_JSON = re.compile(r"\{.*\}", re.S)


def log(message: str) -> None:
    print(f"[0078] {message}", flush=True)


def parse_choices(text: str) -> dict[int, str] | None:
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
    return {
        row["cluster_id"]: str(row.get("choice", ""))
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), int)
    }


def finalize(entity: dict[str, Any], assertions: tuple[str, ...]) -> dict[str, Any]:
    """Serialize one entity: frozen assertions, forced-empty candidates."""
    row = {"text": entity["text"], "type": entity["type"], "position": entity["position"]}
    if entity["type"] in ASSERTION_TYPES:
        row["assertions"] = list(entity.get("assertions", assertions))
    if entity["type"] in CANDIDATE_TYPES:
        row["candidates"] = []
    return row


class MergeModel:
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
        log("Qwen3-8B loaded (greedy)")

    def ask(self, user: str) -> dict[int, str] | None:
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
        return parse_choices(
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
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--expected-documents", type=int, default=100)
    parser.add_argument("--documents", default="", help="comma-separated stems, e.g. 1,10,100")
    args = parser.parse_args(argv)

    if "semantic_s1_0072" in (
        args.seed_run_config.read_text(encoding="utf-8") if args.seed_run_config.is_file() else ""
    ):
        print(
            "BLOCKED: seed deploys the 4B pair; with Qwen3-8B that is 16,369,296,389 "
            "parameters against a 9,000,000,000 cap.",
            file=sys.stderr,
        )
        return 2
    total = budget.assert_within_cap(with_semantic_s1=False)
    log(f"seed OK (E3-only) | deployed parameters {total:,}")

    notes = sorted(p for p in args.input_dir.glob("*") if p.is_file())
    if args.documents:
        wanted = {s.strip() for s in args.documents.split(",") if s.strip()}
        notes = [p for p in notes if p.stem in wanted]
    smoke = bool(args.documents)
    if smoke:
        log(f"SMOKE: {[p.stem for p in notes]} - NOT a submission")
    if not notes:
        print("BLOCKED: no documents selected", file=sys.stderr)
        return 2

    model = MergeModel(args.model_root, args.device, args.max_new_tokens)
    out_dir = args.run_dir / "output_raw_dotless"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stats: Counter[str] = Counter()
    merges: list[dict[str, Any]] = []

    for path in notes:
        note = load_note(path)
        seed_path = args.seed_entities / f"{path.stem}.json"
        proposals = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.is_file() else []
        stats["e3_entities_in"] += len(proposals)

        clusters, untouched = build_merge_clusters(note, proposals)
        stats["eligible_multi_fragment_clusters"] += len(clusters)

        choices: dict[int, str] = {}
        if clusters:
            reply = model.ask(merge_only_prompt(note, clusters))
            if reply is None:
                stats["parse_fallbacks"] += 1
            else:
                choices = reply

        entities = [finalize(e, tuple(e.get("assertions") or [])) for e in untouched]
        for cluster in clusters:
            produced, merged = apply_merge(cluster, choices.get(cluster.cluster_id), note)
            if merged:
                stats["clusters_merged"] += 1
                stats["fragments_removed"] += len(cluster.proposals) - 1
                stats["merged_entities_created"] += 1
                merges.append({"document": path.stem, **cluster.as_dict()})
            else:
                stats["clusters_kept"] += 1
            entities.extend(finalize(e, cluster.assertions) for e in produced)

        entities.sort(key=lambda e: (e["position"][0], e["position"][1], e["type"]))
        stats["entities_out"] += len(entities)
        (out_dir / f"{path.stem}.json").write_text(
            json.dumps(entities, ensure_ascii=False), encoding="utf-8"
        )

    report = {
        "merge_only_version": MERGE_ONLY_VERSION,
        "documents": len(notes),
        **{
            k: stats[k]
            for k in (
                "e3_entities_in",
                "entities_out",
                "eligible_multi_fragment_clusters",
                "clusters_merged",
                "clusters_kept",
                "fragments_removed",
                "merged_entities_created",
                "parse_fallbacks",
            )
        },
        # Structurally zero: these operations do not exist in the 0078 action space.
        "boundary_expansions": 0,
        "single_proposal_changes": 0,
        "type_changes": 0,
        "assertion_changes": 0,
        "only_operation": MERGE_TO_UNION,
        "candidate_policy": "forced_all_null",
        "merges": merges if len(merges) <= 30 else merges[:30],
        "merges_truncated": len(merges) > 30,
        "merges_total": len(merges),
        "model": budget.QWEN3_8B,
        "revision": budget.QWEN3_8B_REVISION,
        "deployed_parameters": total,
        "training_performed": False,
        "submitted": False,
        "runtime_seconds": round(time.time() - started, 1),
        "contains_clinical_text": bool(merges),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "diagnostic-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if len(merges) > 30:
        (args.run_dir / "merges-full.json").write_text(
            json.dumps(merges, indent=2, sort_keys=True), encoding="utf-8"
        )

    log(
        f"in {stats['e3_entities_in']} -> out {stats['entities_out']} | "
        f"eligible {stats['eligible_multi_fragment_clusters']} | "
        f"merged {stats['clusters_merged']} kept {stats['clusters_kept']} | "
        f"fragments removed {stats['fragments_removed']} | "
        f"fallbacks {stats['parse_fallbacks']}"
    )
    for merge in merges[:30]:
        log(
            f"  [{merge['document']}] {merge['original_texts']} -> "
            f"'{merge['union_text']}' {merge['union_offsets']} [{merge['type']}]"
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
