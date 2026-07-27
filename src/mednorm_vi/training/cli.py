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
from .s1_full_training import (
    full_training_output_paths,
    is_supervised_example,
    resolve_s1_best_checkpoint,
    validate_best_checkpoint_only,
    validate_full_training_artifact,
)
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


def run_full_training_validation(args: argparse.Namespace) -> int:
    """Read-only validation of a completed full-training artifact directory.

    Nothing is written, regenerated or copied. Checkpoint tensors are only opened
    when ``--load-checkpoints`` is given, which additionally requires Torch; the
    file-level and manifest checks need neither Torch nor a GPU.
    """
    payloads: dict[str, dict[str, Any]] | None = None
    if args.load_checkpoints:
        import torch  # noqa: PLC0415

        paths = full_training_output_paths(args.artifact_dir)
        payloads = {}
        for name in ("best_checkpoint", "latest_checkpoint"):
            if Path(paths[name]).is_file():
                loaded = torch.load(paths[name], map_location="cpu", weights_only=False)
                # Keep only the metadata; tensors are never retained or copied.
                payloads[name] = {k: v for k, v in loaded.items()
                                  if k not in ("model_state_dict", "optimizer_state_dict",
                                               "scheduler_state_dict")}
                payloads[name].update({k: True for k in loaded
                                       if k.endswith("_state_dict")})
                del loaded

    outcome = validate_full_training_artifact(
        args.artifact_dir,
        expected_checkpoint_sha256={
            "best_checkpoint": args.expected_best_sha256,
            "latest_checkpoint": args.expected_latest_sha256,
        },
        expected_pinned_revision=args.revision,
        checkpoint_payloads=payloads,
    )
    payload = outcome.as_dict()
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not outcome.validated:
        print(f"FULL-TRAINING ARTIFACT VALIDATION FAILED - {len(outcome.failures)} condition(s)")
        for failure in outcome.failures:
            print("  -", failure)
        return 1
    print("FULL-TRAINING ARTIFACT VALIDATED - every condition passed.")
    return 0


def run_best_checkpoint_validation(args: argparse.Namespace) -> int:
    """Validate ONLY the local best checkpoint, read-only, on CPU.

    Deliberately narrower than ``validate-full-training``: it reports
    ``full_artifact_validated: false`` because a lone ``best.pt`` cannot prove the
    complete artifact. Nothing is written and no optimizer state is created.
    """
    location = resolve_s1_best_checkpoint(
        repository_root=args.repository_root, in_colab=args.assume_colab or None)
    checkpoint = Path(args.checkpoint) if args.checkpoint else location.require()

    payload = None
    if not args.skip_payload:
        import torch  # noqa: PLC0415

        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # Metadata plus a boolean marker for the tensors; weights are not copied.
        payload = {k: v for k, v in loaded.items() if not k.endswith("_state_dict")}
        payload.update({k: bool(loaded[k]) for k in loaded if k.endswith("_state_dict")})
        del loaded

    outcome = validate_best_checkpoint_only(
        checkpoint,
        expected_sha256=args.expected_sha256,
        expected_pinned_revision=args.revision,
        expected_epoch=args.expected_epoch,
        expected_global_step=args.expected_global_step,
        payload=payload,
    )
    result = outcome.as_dict() | {"resolution": location.as_dict()}
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not outcome.best_checkpoint_validated:
        for failure in outcome.failures:
            print("  -", failure)
        return 1
    print("BEST CHECKPOINT VALIDATED (full_artifact_validated: false - "
          "run validate-full-training against the complete artifact).")
    return 0


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

    validate = sub.add_parser(
        "validate-full-training",
        help="read-only validation of a completed S1 full-training artifact")
    validate.add_argument("--artifact-dir", required=True)
    validate.add_argument("--revision", required=True,
                          help="pinned immutable 40-hex model revision")
    validate.add_argument("--expected-best-sha256", default="")
    validate.add_argument("--expected-latest-sha256", default="")
    validate.add_argument("--load-checkpoints", action="store_true",
                          help="also verify checkpoint resume metadata (requires Torch)")
    validate.add_argument("--report", default=None)

    best = sub.add_parser(
        "validate-s1-best-checkpoint",
        help="read-only validation of the local S1 best checkpoint alone (CPU)")
    best.add_argument("--checkpoint", default=None,
                      help="override the resolved path entirely")
    best.add_argument("--repository-root", default=None)
    best.add_argument("--revision", required=True)
    best.add_argument("--expected-sha256", default="")
    best.add_argument("--expected-epoch", type=int, required=True)
    best.add_argument("--expected-global-step", type=int, required=True)
    best.add_argument("--assume-colab", action="store_true")
    best.add_argument("--skip-payload", action="store_true",
                      help="hash and locate only; do not open the checkpoint (no Torch)")
    best.add_argument("--report", default=None)

    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(asdict(build_training_plan(artifact_root=args.artifact_root)),
                         sort_keys=True, indent=2))
        return 0
    if args.command == "align-preflight":
        return run_local_alignment_preflight(args)
    if args.command == "validate-full-training":
        return run_full_training_validation(args)
    if args.command == "validate-s1-best-checkpoint":
        return run_best_checkpoint_validation(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
