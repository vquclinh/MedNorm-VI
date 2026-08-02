#!/usr/bin/env python3
"""Variant F: E3-constrained Qwen3-8B verifier + all-null candidates (sprint 0076).

    note + E3 proposals -> ONE Qwen3-8B generation per document
                        -> decisions keyed by proposal id only
                        -> strict projection (text/position from E3)
                        -> candidates = [] for every CHẨN_ĐOÁN / THUỐC

Deployed: ViHealthBERT E3 135,002,117 + Qwen3-8B 8,190,735,360 = 8,325,737,477 (< 9B).
No Embedding-4B, no Reranker-4B, no dense cache, no candidate retrieval. Greedy decoding, no
sampling parameters set. Nothing is trained.
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

from mednorm_vi.reasoner import budget  # noqa: E402
from mednorm_vi.reasoner.prompt import SYSTEM, verify_prompt  # noqa: E402
from mednorm_vi.reasoner.validator import ORGANIZER_TYPES  # noqa: E402
from mednorm_vi.reasoner.verifier import (  # noqa: E402
    FALLBACK_KEEP_E3,
    VerifierDiagnostics,
    apply_decisions,
)

_JSON = re.compile(r"\{.*\}", re.S)


def log(message: str) -> None:
    print(f"[0076] {message}", flush=True)


def parse_decisions(text: str) -> list[dict[str, Any]] | None:
    """Decisions, or None when the reply is unusable (triggers the documented fallback)."""
    match = _JSON.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    decisions = payload.get("decisions") if isinstance(payload, dict) else None
    return decisions if isinstance(decisions, list) else None


class Verifier:
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
        # Greedy decoding: clear sampling defaults so transformers does not warn about
        # temperature/top_p/top_k being set while do_sample=False.
        config = self._model.generation_config
        config.do_sample = False
        for name in ("temperature", "top_p", "top_k"):
            if hasattr(config, name):
                setattr(config, name, None)
        log("Qwen3-8B loaded (greedy, no sampling parameters)")

    def ask(self, user: str) -> list[dict[str, Any]] | None:
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
        reply = self._tokenizer.decode(
            generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return parse_decisions(reply)


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
    parser.add_argument(
        "--seed-entities",
        type=Path,
        required=True,
        help="E3-only organizer JSON directory (variant F verifies these)",
    )
    parser.add_argument(
        "--seed-run-config",
        type=Path,
        required=True,
        help="the pipeline YAML that produced the seed; must be E3-only",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--expected-documents", type=int, default=100)
    parser.add_argument(
        "--limit-documents",
        type=int,
        default=0,
        help="smoke test only; the document-count check is skipped",
    )
    args = parser.parse_args(argv)

    seed_text = (
        args.seed_run_config.read_text(encoding="utf-8") if args.seed_run_config.is_file() else ""
    )
    if "semantic_s1_0072" in seed_text:
        print(
            "BLOCKED: the seed uses linker_backend=semantic_s1_0072, which deploys "
            "Embedding-4B + Reranker-4B. With Qwen3-8B that is 16,369,296,389 parameters "
            "against a 9,000,000,000 cap. Seed variant F from an E3-only profile.",
            file=sys.stderr,
        )
        return 2
    total = budget.assert_within_cap(with_semantic_s1=False)
    log(f"seed provenance OK (E3-only) | deployed parameters {total:,}")

    notes = sorted(p for p in args.input_dir.glob("*") if p.is_file())
    smoke = args.limit_documents > 0
    if smoke:
        notes = notes[: args.limit_documents]
        log(f"SMOKE TEST: {len(notes)} documents only - NOT a submission")
    if not notes:
        print(f"BLOCKED: no documents in {args.input_dir}", file=sys.stderr)
        return 2

    verifier = Verifier(args.model_root, args.device, args.max_new_tokens)
    diagnostics = VerifierDiagnostics()
    out_dir = args.run_dir / "output_raw_dotless"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for position, path in enumerate(notes, start=1):
        seed_path = args.seed_entities / f"{path.stem}.json"
        proposals = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.is_file() else []
        note = load_note(path)
        decisions = verifier.ask(verify_prompt(note, proposals)) if proposals else []
        entities = apply_decisions(proposals, decisions, diagnostics)
        (out_dir / f"{path.stem}.json").write_text(
            json.dumps(entities, ensure_ascii=False), encoding="utf-8"
        )
        if position % 5 == 0 or position == len(notes):
            elapsed = time.time() - started
            rate = position / elapsed if elapsed else 0.0
            log(
                f"  {position}/{len(notes)} docs "
                f"({rate * 60:.1f}/min, ETA {(len(notes) - position) / rate / 60:.1f} min)"
                if rate
                else f"  {position}/{len(notes)} docs"
            )

    produced = sorted(out_dir.glob("*.json"))
    report = {
        **diagnostics.as_dict(),
        "model": budget.QWEN3_8B,
        "revision": budget.QWEN3_8B_REVISION,
        "deployed_parameters": total,
        "parameter_cap": budget.PARAMETER_CAP,
        "parse_fallback_policy": FALLBACK_KEEP_E3,
        "allowed_types": list(ORGANIZER_TYPES),
        "free_form_entity_discovery": False,
        "training_performed": False,
        "submitted": False,
        "runtime_seconds": round(time.time() - started, 1),
        "contains_clinical_text": False,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "diagnostic-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    log(
        f"proposals {diagnostics.proposals_in} -> kept {diagnostics.kept}, "
        f"dropped {diagnostics.dropped}, type changes {diagnostics.type_changes}, "
        f"assertion changes {diagnostics.assertion_changes}, "
        f"fallbacks {diagnostics.parse_fallbacks}"
    )
    if smoke:
        log("smoke complete")
        return 0
    if len(produced) != args.expected_documents:
        print(
            f"BLOCKED: produced {len(produced)}, expected {args.expected_documents}",
            file=sys.stderr,
        )
        return 3
    log(f"raw output -> {out_dir}. NEXT: dotted derivation, then package. NOT submitted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
