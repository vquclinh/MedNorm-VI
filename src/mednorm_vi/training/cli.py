"""Training workflow CLI for dry-run planning and readiness checks."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .phobert_alignment import (
    map_segmented_words,
    resolve_segmented_text,
    segmented_text_to_words,
    verify_tokenizer_equivalence,
)
from .s1_full_training import is_supervised_example
from .s1_mention_smoke import (
    encode_mention_example_slow,
    iter_jsonl,
    load_coverage,
    load_governed_exclusions,
    loss_mask_for_example,
    run_alignment_preflight,
)
from .stages import build_training_plan

DEFAULT_CORPUS_DIR = "data/derived/training_corpora/mednorm_vi_training_v1"
DEFAULT_MODEL_ID = "demdecuong/vihealthbert-base-word"
DEFAULT_EXCLUSIONS = "configs/training/s1_governed_exclusions.yaml"
DEFAULT_REPORT = "reports/local_alignment_preflight/alignment_preflight.json"
PREFLIGHT_SPLITS = ("train", "validation")


def _build_segmenter(vncorenlp_dir: str) -> Any:
    """Real VnCoreNLP RDRSegmenter. CPU only; no model weights are involved.

    ``py_vncorenlp`` chdirs into its resource directory, so the caller's working
    directory is restored after every call.
    """
    py_vncorenlp: Any = importlib.import_module("py_vncorenlp")

    resource_dir = str(Path(vncorenlp_dir).expanduser().resolve())
    working_directory = os.getcwd()
    if not any(Path(resource_dir).glob("*.jar")):
        Path(resource_dir).mkdir(parents=True, exist_ok=True)
        try:
            py_vncorenlp.download_model(save_dir=resource_dir)
        finally:
            os.chdir(working_directory)
    try:
        segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=resource_dir)
    finally:
        os.chdir(working_directory)

    def segment(text: str) -> str:
        try:
            segments = segmenter.word_segment(text)
        finally:
            os.chdir(working_directory)
        if not segments:
            raise ValueError("RDRSegmenter returned no segments")
        return " ".join(segments)

    return segment


def _load_slow_tokenizer(model_id: str, revision: str, cache_dir: str | None) -> Any:
    """The real slow PhobertTokenizer. Tokenizer files only - never weights."""
    transformers_module: Any = importlib.import_module("transformers")
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(
        model_id, revision=revision, use_fast=False,
        **({"cache_dir": cache_dir} if cache_dir else {}))
    if getattr(tokenizer, "is_fast", False):
        raise ValueError("expected the slow PhobertTokenizer; got a fast tokenizer")
    return tokenizer


def run_local_alignment_preflight(args: argparse.Namespace) -> int:
    """Full supervised train+validation alignment preflight, CPU only.

    Segmentation, character alignment, mention encoding and tokenizer equivalence
    over every supervised example. No model is constructed, no forward or backward
    pass runs, and no checkpoint is read or written.
    """
    corpus_dir = Path(args.corpus_dir)
    missing = [
        str(corpus_dir / "splits" / f"{split}.jsonl") for split in PREFLIGHT_SPLITS
        if not (corpus_dir / "splits" / f"{split}.jsonl").is_file()
    ]
    if not (corpus_dir / "manifests" / "annotation_coverage.json").is_file():
        missing.append(str(corpus_dir / "manifests" / "annotation_coverage.json"))
    if missing:
        print(json.dumps({"error": "missing required corpus file(s)",
                          "missing_paths": missing}, indent=2))
        return 2

    coverage = load_coverage(corpus_dir)
    exclusions = load_governed_exclusions(args.exclusions)
    segment_with_vncorenlp = _build_segmenter(args.vncorenlp_dir)
    tokenizer = _load_slow_tokenizer(args.model_id, args.revision, args.tokenizer_cache_dir)

    def segmented_form(text: str) -> str:
        segmented, _source = resolve_segmented_text(text, segment_with_vncorenlp)
        return segmented

    def encode(row: Mapping[str, Any]) -> Any:
        return encode_mention_example_slow(
            row, tokenizer, coverage_by_source=coverage,
            max_length=args.max_sequence_length,
            segmented_text=segmented_form(str(row["text"])))

    def verify(row: Mapping[str, Any]) -> None:
        words = map_segmented_words(
            str(row["text"]), segmented_text_to_words(segmented_form(str(row["text"]))))
        verify_tokenizer_equivalence(words, tokenizer)

    def supervised(row: Mapping[str, Any]) -> bool:
        return is_supervised_example(loss_mask_for_example(row, coverage))

    def progress(split: str, index: int, considered: int) -> None:
        print(f"  {split}: row {index} ({considered} considered, "
              f"{time.time() - started:.0f}s)", flush=True)

    started = time.time()
    report, _features = run_alignment_preflight(
        {split: iter_jsonl(corpus_dir / "splits" / f"{split}.jsonl")
         for split in PREFLIGHT_SPLITS},
        encode=encode, verify_equivalence=verify,
        governed_exclusions=exclusions, supervised=supervised, progress=progress)

    payload = report.as_dict()
    payload["runtime"] = {
        "elapsed_seconds": round(time.time() - started, 1),
        "corpus_dir": str(corpus_dir),
        "model_id": args.model_id,
        "pinned_revision": args.revision,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "segmenter": "VnCoreNLP RDRSegmenter (py_vncorenlp)",
        "max_sequence_length": args.max_sequence_length,
        "model_constructed": False,
        "forward_or_backward_pass": False,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("unalignable_examples", "governed_exclusions")},
                     indent=2, sort_keys=True))
    if payload["unalignable_examples"]:
        print("UNEXPECTED unalignable examples (privacy-safe diagnostics):")
        for entry in payload["unalignable_examples"][:40]:
            print("  ", json.dumps(entry, sort_keys=True))
    print(f"report written to {destination}")
    return 0 if report.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="emit deterministic S0-S6 training plan")
    plan.add_argument("--artifact-root", default="models/checkpoints/full_v1")

    preflight = sub.add_parser(
        "align-preflight",
        help="full supervised train+validation S1 alignment preflight (CPU only)")
    preflight.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    preflight.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    preflight.add_argument("--revision", required=True,
                           help="pinned immutable 40-hex model revision")
    preflight.add_argument("--vncorenlp-dir", required=True,
                           help="untracked directory holding VnCoreNLP resources")
    preflight.add_argument("--tokenizer-cache-dir", default=None)
    preflight.add_argument("--exclusions", default=DEFAULT_EXCLUSIONS)
    preflight.add_argument("--max-sequence-length", type=int, default=256)
    preflight.add_argument("--report", default=DEFAULT_REPORT)

    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(asdict(build_training_plan(artifact_root=args.artifact_root)),
                         sort_keys=True, indent=2))
        return 0
    if args.command == "align-preflight":
        return run_local_alignment_preflight(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
