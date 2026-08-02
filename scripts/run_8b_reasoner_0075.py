#!/usr/bin/env python3
"""Constrained 8B unified reasoner -> organizer JSON (sprint 0075, variant E_full_8b).

Zero-shot. No training. The model proposes; `reasoner.validator` refuses anything it cannot
prove against the source text and the governed KB, so no invented span, type or code can reach
the output.

Configurations (`--config`):
    full_8b   8B for entities + assertions + candidate selection
    entity_8b 8B for entities + assertions only; candidates left empty for a later linker
    cand_8b   supplied entities/assertions; 8B selects candidates only

Deployed: ViHealthBERT E3 + Qwen3-8B = 8,325,737,477 (< 9B). The Audit-0073 embedding and
reranker pair CANNOT co-deploy; `reasoner.budget` refuses that combination.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.reasoner import budget, prompt  # noqa: E402
from mednorm_vi.reasoner.pool import build_pool  # noqa: E402
from mednorm_vi.reasoner.validator import (  # noqa: E402
    ASSERTION_KEYS,
    CANDIDATE_TYPES,
    validate,
)

CONFIGS = ("full_8b", "entity_8b", "cand_8b")
_JSON = re.compile(r"\{.*\}", re.S)


def log(message: str) -> None:
    print(f"[0075] {message}", flush=True)


def parse_json(text: str) -> dict[str, Any]:
    """Best-effort strict-JSON extraction. A malformed reply yields {} and is dropped."""
    match = _JSON.search(text or "")
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


class Reasoner:
    """Qwen3-8B, greedy, loaded once. Weights load on first use, never at import."""

    def __init__(self, model_root: Path, device: str, max_new_tokens: int) -> None:
        self.model_root = model_root
        self.device = device
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
        log(f"Qwen3-8B loaded from {source}")

    def ask(self, user: str) -> dict[str, Any]:
        self._load()
        import torch

        messages = [
            {"role": "system", "content": prompt.SYSTEM},
            {"role": "user", "content": user},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = self._tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        reply = self._tokenizer.decode(
            generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return parse_json(reply)


def load_seed(path: Path) -> list[dict[str, Any]]:
    """Read a prior run's organizer JSON as reasoner proposals.

    Organizer output stores `assertions` as a LIST of set flags; proposals carry a dict of
    booleans. Converting here lets variant D reuse the historical E3 entity/assertion output
    verbatim - the whole point of D is that those stay untouched.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        flags = row.get("assertions")
        if isinstance(flags, list):
            flags = {key: key in flags for key in ASSERTION_KEYS}
        out.append({
            "text": row.get("text", ""),
            "type": row.get("type", ""),
            "assertions": flags or {},
        })
    return out


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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help=f"local Qwen3-8B directory (revision {budget.QWEN3_8B_REVISION})",
    )
    parser.add_argument("--config", choices=CONFIGS, default="full_8b")
    parser.add_argument(
        "--icd-v3", type=Path, default=REPO / "indices/candidate/icd10_vi/competition-v3/index.json"
    )
    parser.add_argument(
        "--icd-v41",
        type=Path,
        default=REPO / "indices/candidate/icd10_vi/competition-v4.1/index.json",
    )
    parser.add_argument(
        "--rxnorm", type=Path, default=REPO / "indices/candidate/rxnorm/competition-v3/index.json"
    )
    parser.add_argument(
        "--structured",
        type=Path,
        default=REPO / "artifacts/rxnorm_structured/0074/structured.jsonl",
    )
    parser.add_argument(
        "--seed-entities",
        type=Path,
        default=None,
        help="optional directory of existing per-document entity JSON",
    )
    parser.add_argument("--pool-limit", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-documents", type=int, default=100)
    args = parser.parse_args(argv)

    if args.config == "cand_8b" and not args.seed_entities:
        print("BLOCKED: --config cand_8b needs --seed-entities pointing at a prior run's "
              "organizer JSON (variant D keeps the historical E3 entities/assertions).",
              file=sys.stderr)
        return 2

    total_parameters = budget.assert_within_cap(with_semantic_s1=False)
    log(f"deployed parameters {total_parameters:,} (cap {budget.PARAMETER_CAP:,})")

    blockers = [
        f"missing {label}: {path}"
        for label, path in (
            ("input dir", args.input_dir),
            ("Qwen3-8B", args.model_root / "config.json"),
            ("ICD v3", args.icd_v3),
            ("ICD v4.1", args.icd_v41),
            ("RxNorm", args.rxnorm),
        )
        if not path.exists()
    ]
    if blockers:
        print("BLOCKED:", file=sys.stderr)
        for blocker in blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 2

    from mednorm_vi.kb.indexing.retrieval import load_index
    from mednorm_vi.kb.rxnorm.structured import StructuredDrug

    icd_indexes = [("v3", load_index(args.icd_v3)), ("v41", load_index(args.icd_v41))]
    rx_indexes = [("rx", load_index(args.rxnorm))]
    structured: dict[str, StructuredDrug] = {}
    if args.structured.is_file():
        with args.structured.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    drug = StructuredDrug.from_dict(json.loads(line))
                    structured[drug.rxcui] = drug
        log(f"structured RxNorm records {len(structured):,}")

    reasoner = Reasoner(args.model_root, args.device, args.max_new_tokens)
    notes = sorted(args.input_dir.glob("*"))
    notes = [p for p in notes if p.is_file()]
    log(f"documents {len(notes)} | config {args.config}")

    out_dir = args.run_dir / "output_raw_dotless"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    aggregate: dict[str, Any] = {"accepted": 0, "rejected": {}}

    for position, path in enumerate(notes, start=1):
        note = load_note(path)
        seed: list[dict[str, Any]] = []
        if args.seed_entities:
            seed_path = args.seed_entities / f"{path.stem}.json"
            if seed_path.is_file():
                seed = load_seed(seed_path)

        if args.config == "cand_8b":
            proposals = [dict(e) for e in seed]
        else:
            reply = reasoner.ask(prompt.entity_prompt(note, seed))
            proposals = [
                {"text": e.get("text", ""), "type": e.get("type", "")}
                for e in (reply.get("entities") or [])
                if isinstance(e, dict)
            ]
            listed = [{"text": p["text"], "type": p["type"]} for p in proposals]
            if listed:
                flags = reasoner.ask(prompt.assertion_prompt(note, listed))
                for row in flags.get("assertions") or []:
                    index = row.get("index")
                    if isinstance(index, int) and 0 <= index < len(proposals):
                        proposals[index]["assertions"] = {
                            key: bool(row.get(key, False)) for key in ASSERTION_KEYS
                        }

        governed: dict[str, set[str]] = {}
        if args.config in ("full_8b", "cand_8b"):
            for index, proposal in enumerate(proposals):
                if proposal.get("type") not in CANDIDATE_TYPES:
                    continue
                is_icd = proposal["type"] == "CHẨN_ĐOÁN"
                pool = build_pool(
                    proposal.get("text", ""),
                    icd_indexes if is_icd else rx_indexes,
                    limit=args.pool_limit,
                    structured=None if is_icd else structured,
                    dotted=is_icd,
                )
                key = str(index)
                proposal["pool_key"] = key
                governed[key] = {c.display_code for c in pool}
                if not pool:
                    proposal["candidates"] = []
                    continue
                chosen = reasoner.ask(
                    prompt.candidate_prompt(note, proposal["text"], proposal["type"], pool)
                )
                picked = [
                    pool[i].display_code
                    for i in (chosen.get("selected") or [])
                    if isinstance(i, int) and 0 <= i < len(pool)
                ]
                proposal["candidates"] = picked

        entities, report = validate(note, proposals, governed)
        aggregate["accepted"] += report.accepted
        for reason, count in report.rejected.items():
            aggregate["rejected"][reason] = aggregate["rejected"].get(reason, 0) + count
        (out_dir / f"{path.stem}.json").write_text(
            json.dumps([e.as_organizer_json() for e in entities], ensure_ascii=False),
            encoding="utf-8",
        )
        if position % 5 == 0 or position == len(notes):
            elapsed = time.time() - started
            rate = position / elapsed if elapsed else 0.0
            log(
                f"  {position}/{len(notes)} docs ({rate * 60:.1f} docs/min, "
                f"ETA {(len(notes) - position) / rate / 60:.1f} min)"
                if rate
                else f"  {position}/{len(notes)} docs"
            )

    produced = sorted(out_dir.glob("*.json"))
    if len(produced) != args.expected_documents:
        print(
            f"BLOCKED: produced {len(produced)} documents, expected {args.expected_documents}",
            file=sys.stderr,
        )
        return 3

    (args.run_dir / "model-manifest.json").write_text(
        json.dumps(budget.manifest(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.run_dir / "validation-report.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.run_dir / "diagnostic-summary.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "documents": len(produced),
                "entities_accepted": aggregate["accepted"],
                "rejected": aggregate["rejected"],
                "runtime_seconds": round(time.time() - started, 1),
                "model": budget.QWEN3_8B,
                "revision": budget.QWEN3_8B_REVISION,
                "training_performed": False,
                "submitted": False,
                "contains_clinical_text": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    log(f"raw output -> {out_dir}")
    log(f"entities accepted {aggregate['accepted']:,}; rejected {aggregate['rejected']}")
    log("NEXT: apply the governed dotted derivation, then package. NOT submitted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
