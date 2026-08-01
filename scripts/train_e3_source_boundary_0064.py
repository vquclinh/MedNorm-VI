#!/usr/bin/env python3
"""Source-balanced and boundary-weighted refinement of the E3 head (Audit 0064).

Forked from `train_e3_boundary_refinement_0062.py`, which stays untouched because Audit
0062 cites it as the exact provenance of the active checkpoint. Two additions, each
independently switchable so R4 and R5 stay attributable:

* ``--boundary-weight W`` multiplies the loss on tokens at the FIRST or LAST position of
  a gold mention run. Evidence (Audit 0064 §3): 298 boundary errors remain - 192 right,
  91 left, 15 both - and right-boundary error still outnumbers left 2:1.
* ``--source-balanced`` draws each epoch uniformly across the governed source datasets
  instead of in corpus proportion. In THIS corpus the sources are type-disjoint -
  vimedner carries all DIAGNOSIS/SYMPTOM supervision at 24.4% of training, while
  vietmed_ner and vimq carry only MEDICATION at 75.6% - so source balancing is
  effectively type rebalancing, and the audit says so rather than letting the label
  "source-balanced" imply something narrower.

The sampler is uniform BY SOURCE, not matched to the validation mixture. Matching
validation would be fitting the evaluation set, which Audit 0062 §4.6 explicitly declined
to do; uniform is a principled prior that happens to favour the under-supervised types.

Original contract (unchanged): governed splits resolved by SHA-256, `internal_test`
refused, source checkpoint hashed before and after, architecture rebuilt from the pinned
config and loaded strict=True, selection on EXACT character-span + type F1.

Runs from a clean clone with no notebook and no notebook-only state. Every asset is
resolved by digest and every precondition fails closed:

* the governed splits are located by authoritative SHA-256, never by name or path;
* ``internal_test`` is refused by :mod:`mednorm_vi.training.governed_splits`;
* the source checkpoint is hashed before a single byte is deserialized;
* the architecture is rebuilt from the pinned config and loaded ``strict=True``;
* the validated ``best.pt`` is opened read-only and never written to.

**What this script does NOT do.** It does not reuse
``s1_full_training.FullTrainingConfig``. That contract refuses any initializer other
than ``pretrained_base`` on purpose - for a from-scratch S1 run, a checkpoint
initializer would launder the one-step smoke artifact into a full run. Refinement is
the case that rule was not written for, so it carries its own explicit contract here
rather than a weakened version of that one.

**Selection metric.** Epoch selection uses the EXACT character-span + type evaluator
(``evaluation.exact_mention``) decoded through the production rule
(``mention_factory.neural.decoding.decode_type_runs``). The original run selected on
``validation_span_micro_f1``, a token-run proxy that can rise while exact character
spans get worse. Brief §7: never select on token-level accuracy.

Usage::

    python scripts/train_e3_boundary_refinement_0062.py \
        --config configs/training/e3_boundary_refinement_0062.yaml \
        --run-id R1_lowlr --output-root checkpoint/experiments/0062_e3_boundary_refinement \
        --confirm "RUN E3 BOUNDARY REFINEMENT"

``--focal-alpha`` is the only recipe override, so R1 and R2 differ by one scalar and
the comparison attributes cleanly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.evaluation.exact_mention import Mention, evaluate_examples  # noqa: E402
from mednorm_vi.mention_factory.neural.decoding import decode_type_runs  # noqa: E402
from mednorm_vi.mention_factory.neural.runtime import encode_for_inference  # noqa: E402
from mednorm_vi.training.governed_splits import resolve_governed_splits  # noqa: E402
from mednorm_vi.training.s1_full_training import is_supervised_example  # noqa: E402
from mednorm_vi.training.s1_mention_smoke import (  # noqa: E402
    ENTITY_TYPE_ORDER,
    encode_mention_example_slow,
    iter_jsonl,
    load_coverage,
    loss_mask_for_example,
    pad_encoded_features,
)

CONTRACT_VERSION = "e3-source-boundary-refinement-0064-v1"
BEST_METRIC_KEY = "exact_span_type_micro_f1"
CHECKPOINT_MODE = "FULL_TRAINING"


class RefinementError(RuntimeError):
    """Any violated precondition. Every one of these stops the run."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name) or {}
        if not isinstance(value, dict):
            raise RefinementError(f"config section {name!r} must be a mapping")
        return value

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: Path) -> RefinementConfig:
    import yaml

    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise RefinementError("refinement config must be a mapping")
    config = RefinementConfig(doc)
    model = config.section("model")
    if str(model.get("initialize_from")) != "refinement_checkpoint":
        raise RefinementError(
            "this script only performs continued refinement; "
            f"initialize_from is {model.get('initialize_from')!r}"
        )
    for required in ("source_checkpoint", "source_checkpoint_sha256", "pinned_revision"):
        if not str(model.get(required, "")).strip():
            raise RefinementError(f"model.{required} is required and must not be empty")
    if len(str(model["pinned_revision"])) != 40:
        raise RefinementError("model.pinned_revision must be an immutable 40-hex commit")
    return config


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def set_determinism(seed: int) -> None:
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # cuDNN NER kernels have no det. path
    torch.backends.cudnn.benchmark = False


# --- data ---------------------------------------------------------------------


def _compact(feature: dict[str, Any], *, keep_offsets: bool) -> dict[str, Any]:
    """Store one encoded example as small arrays instead of Python int lists.

    33,826 training examples held as nested lists is roughly 2 GB of boxed ints,
    which does not fit the memory budget this milestone runs under. The arrays hold
    exactly the same values; ``pad_encoded_features`` still does the padding, so the
    supervision semantics are untouched.
    """
    import numpy

    compact = {
        "example_id": str(feature.get("example_id", "")),
        "source_dataset": str(feature.get("source_dataset", "")),
        "input_ids": numpy.asarray(feature["input_ids"], dtype=numpy.int32),
        "labels": numpy.asarray(feature["labels"], dtype=numpy.int8),
        "boundary": boundary_mask_from_labels(feature["labels"]),
        "label_mask": numpy.asarray(feature["label_mask"], dtype=numpy.int8),
        "attention_mask": numpy.asarray(feature["attention_mask"], dtype=numpy.int8),
    }
    if keep_offsets:
        compact["offset_mapping"] = [
            tuple(int(v) for v in pair) for pair in feature["offset_mapping"]
        ]
    return compact


def _expand(feature: dict[str, Any]) -> dict[str, Any]:
    """Back to the plain-list shape ``pad_encoded_features`` is specified against."""
    return {
        "example_id": feature["example_id"],
        "input_ids": feature["input_ids"].tolist(),
        "attention_mask": feature["attention_mask"].tolist(),
        "labels": feature["labels"].tolist(),
        "label_mask": feature["label_mask"].tolist(),
        "boundary": feature["boundary"].tolist(),
    }


def boundary_mask_from_labels(labels: Any) -> Any:
    """1 at the first and last token of every contiguous positive run, per type.

    Derived from the labels themselves, so it needs no extra annotation and cannot drift
    from the supervision it is weighting. A single-token entity is both its own first and
    last token and is marked once.
    """
    import numpy

    array = numpy.asarray(labels, dtype=numpy.int8)
    if array.ndim != 2:
        raise RefinementError("labels must be [tokens, types]")
    previous = numpy.vstack([numpy.zeros((1, array.shape[1]), dtype=numpy.int8), array[:-1]])
    following = numpy.vstack([array[1:], numpy.zeros((1, array.shape[1]), dtype=numpy.int8)])
    starts = (array == 1) & (previous == 0)
    ends = (array == 1) & (following == 0)
    return (starts | ends).astype(numpy.int8)


def build_features(
    split_path: Path,
    tokenizer: Any,
    *,
    coverage: Any,
    max_length: int,
    supervised_only: bool,
    label: str,
    keep_offsets: bool = False,
) -> list[dict[str, Any]]:
    """Encode a governed split with the SAME contracts the original run used."""
    features: list[dict[str, Any]] = []
    skipped = 0
    for index, row in enumerate(iter_jsonl(split_path), start=1):
        if supervised_only and not is_supervised_example(loss_mask_for_example(row, coverage)):
            skipped += 1
            continue
        try:
            feature = encode_mention_example_slow(
                row, tokenizer, coverage_by_source=coverage, max_length=max_length
            )
        except Exception as exc:  # noqa: BLE001 - one bad row must not be silent
            raise RefinementError(
                f"{label}: failed to encode example {index} "
                f"({type(exc).__name__}); refusing to train on a partial corpus"
            ) from exc
        feature["example_id"] = str(row.get("example_id", ""))
        feature["source_dataset"] = str(row.get("source_dataset", ""))
        features.append(_compact(feature, keep_offsets=keep_offsets))
        if index % 5000 == 0:
            print(f"  [{label}] encoded {index} rows -> {len(features)} kept", flush=True)
    print(f"  [{label}] {len(features)} features ({skipped} unsupervised skipped)", flush=True)
    if not features:
        raise RefinementError(f"{label}: no usable features")
    return features


def batches(
    features: list[dict[str, Any]],
    size: int,
    *,
    pad_token_id: int,
    shuffle_rng,
    order: list[int] | None = None,
):
    """Collate a batch, padding the boundary mask alongside the governed fields.

    `pad_encoded_features` is the governed collator and pads exactly the fields it is
    specified against; it ignores unknown keys. The boundary mask is therefore padded here
    rather than by widening that contract, so supervision semantics stay untouched.
    """
    if order is None:
        order = list(range(len(features)))
        if shuffle_rng is not None:
            shuffle_rng.shuffle(order)
    for start in range(0, len(order), size):
        picked = [features[i] for i in order[start : start + size]]
        chunk = [_expand(f) for f in picked]
        batch = pad_encoded_features(chunk, pad_token_id=pad_token_id)
        width = len(batch["input_ids"][0])
        zero_row = [0] * len(ENTITY_TYPE_ORDER)
        batch["boundary"] = [
            list(f["boundary"].tolist())
            + [list(zero_row) for _ in range(width - len(f["boundary"]))]
            for f in picked
        ]
        batch["sources"] = [f["source_dataset"] for f in picked]
        yield batch


def source_balanced_order(features: list[dict[str, Any]], rng: random.Random) -> list[int]:
    """One epoch of indices drawn UNIFORMLY across source datasets.

    Epoch length is unchanged, so the optimizer-step schedule and the effective batch size
    stay comparable to Audit 0062 - only the mixture moves. Each source keeps its own
    shuffled queue and is refilled when exhausted, so a small source is oversampled rather
    than a large one being discarded: no governed training example is thrown away.
    """
    pools: dict[str, list[int]] = {}
    for index, feature in enumerate(features):
        pools.setdefault(feature["source_dataset"], []).append(index)
    names = sorted(pools)
    queues: dict[str, list[int]] = {}
    for name in names:
        queue = list(pools[name])
        rng.shuffle(queue)
        queues[name] = queue
    order: list[int] = []
    position = 0
    while len(order) < len(features):
        name = names[position % len(names)]
        position += 1
        if not queues[name]:
            queues[name] = list(pools[name])
            rng.shuffle(queues[name])
        order.append(queues[name].pop())
    return order


# --- evaluation ---------------------------------------------------------------


def evaluate_exact(
    model: Any,
    torch: Any,
    encoded: list[tuple[str, str, Any]],
    examples: list[dict],
    *,
    pad_token_id: int,
    threshold: float,
    batch_size: int,
    amp_dtype: Any,
    device: Any,
) -> dict[str, Any]:
    """Exact character-span + type metrics through the production decoder.

    ``encoded`` comes from ``neural.runtime.encode_for_inference`` - the SAME encoding
    the shipped runtime uses - so a validation number here means the same thing as a
    number from ``scripts/evaluate_e3_checkpoint_0062.py``. Building it from the
    training encoder instead would measure a path that never runs in production.
    """
    model.eval()
    predictions: dict[str, list[Mention]] = {}
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            chunk = encoded[start : start + batch_size]
            width = max(len(item[2].input_ids) for item in chunk)
            ids = torch.tensor(
                [
                    list(i[2].input_ids) + [pad_token_id] * (width - len(i[2].input_ids))
                    for i in chunk
                ],
                dtype=torch.long,
                device=device,
            )
            attention = torch.tensor(
                [[1] * len(i[2].input_ids) + [0] * (width - len(i[2].input_ids)) for i in chunk],
                dtype=torch.long,
                device=device,
            )
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(ids, attention)
            positives = (torch.sigmoid(logits.float()) > threshold).int().tolist()
            for row, (example_id, text, encoding) in enumerate(chunk):
                length = len(encoding.input_ids)
                spans = decode_type_runs(
                    text, list(encoding.offsets), positives[row][:length], ENTITY_TYPE_ORDER
                )
                predictions[example_id] = [
                    Mention(s.start, s.end, s.entity_type, s.text) for s in spans
                ]
    model.train()
    report = evaluate_examples(examples, predictions=predictions, label="e3_refinement_validation")
    micro = report["micro"]
    by_type = report.get("by_type", {})
    return {
        BEST_METRIC_KEY: micro["f1"],
        "exact_span_type_micro_precision": micro["precision"],
        "exact_span_type_micro_recall": micro["recall"],
        "true_positive": micro["true_positive"],
        "false_positive": micro["false_positive"],
        "false_negative": micro["false_negative"],
        "by_type": {
            k: {m: v[m] for m in ("precision", "recall", "f1")} for k, v in by_type.items()
        },
        "error_categories": report.get("error_categories", {}),
        "prediction_count": sum(len(v) for v in predictions.values()),
    }


# --- main ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--confirm", default="", help="the config's confirmation phrase")
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=1.0,
        help="loss multiplier on tokens at the first/last position of a gold run (1.0 = off)",
    )
    parser.add_argument(
        "--source-balanced",
        action="store_true",
        help="draw each epoch uniformly across governed source datasets",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=0,
        help="0 = all; a positive value is a smoke run and is recorded as such",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    expected_phrase = str(config.raw.get("confirmation_phrase", ""))
    if expected_phrase and args.confirm != expected_phrase:
        raise RefinementError(
            f"refusing to start a long GPU run without confirmation; "
            f'pass --confirm "{expected_phrase}"'
        )

    model_cfg = config.section("model")
    data_cfg = config.section("data")
    opt_cfg = config.section("optimization")
    loss_cfg = config.section("loss")
    eval_cfg = config.section("evaluation")

    seed = int(args.seed if args.seed is not None else config.raw.get("seed", 0))
    focal_alpha = float(
        args.focal_alpha if args.focal_alpha is not None else loss_cfg.get("focal_alpha", 0.25)
    )
    learning_rate = float(
        args.learning_rate if args.learning_rate is not None else opt_cfg["learning_rate"]
    )
    head_lr = float(opt_cfg["head_learning_rate"])
    if args.learning_rate is not None:
        head_lr = learning_rate * (
            float(opt_cfg["head_learning_rate"]) / float(opt_cfg["learning_rate"])
        )
    num_epochs = int(args.epochs if args.epochs is not None else opt_cfg["num_epochs"])
    max_length = int(data_cfg["max_sequence_length"])
    threshold = float(loss_cfg["decision_threshold"])

    out_dir = Path(args.output_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "training_history.jsonl"

    print(f"=== E3 boundary refinement | run {args.run_id} ===", flush=True)
    print(f"contract {CONTRACT_VERSION}  config_sha256 {config.sha256}", flush=True)

    # --- assets, all by digest ------------------------------------------------
    corpus_dir = args.corpus_dir or (REPO / str(data_cfg["corpus_dir_relative"]))
    splits = resolve_governed_splits([corpus_dir])
    print(f"train      {splits['train'].path} sha256={splits['train'].sha256}", flush=True)
    print(
        f"validation {splits['validation'].path} sha256={splits['validation'].sha256}", flush=True
    )

    source_checkpoint = REPO / str(model_cfg["source_checkpoint"])
    if not source_checkpoint.is_file():
        raise RefinementError(f"source checkpoint not found: {source_checkpoint}")
    source_sha = sha256_file(source_checkpoint)
    if source_sha != str(model_cfg["source_checkpoint_sha256"]).strip().lower():
        raise RefinementError(
            f"source checkpoint digest mismatch: recomputed {source_sha}, "
            f"config pins {model_cfg['source_checkpoint_sha256']}"
        )
    print(f"source checkpoint verified {source_sha}", flush=True)

    import torch
    import transformers

    set_determinism(seed)
    if not torch.cuda.is_available():
        raise RefinementError("refinement requires a CUDA device")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    supports_bf16 = bool(torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    if str(opt_cfg.get("mixed_precision", "auto")) == "none":
        amp_dtype = None
    print(
        f"device {torch.cuda.get_device_name(0)} "
        f"{properties.total_memory / 1e9:.1f} GB  amp={amp_dtype}",
        flush=True,
    )

    pinned = str(model_cfg["pinned_revision"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_cfg["hf_model_id"]), revision=pinned, use_fast=False, local_files_only=True
    )
    if bool(getattr(tokenizer, "is_fast", False)):
        raise RefinementError("expected the slow PhobertTokenizer; a fast one changes alignment")
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 1) or 1)

    coverage = load_coverage(corpus_dir)
    print("encoding governed splits (slow tokenizer, original alignment contract)", flush=True)
    train_features = build_features(
        splits["train"].path,
        tokenizer,
        coverage=coverage,
        max_length=max_length,
        supervised_only=bool(data_cfg.get("filter_unsupervised_examples", True)),
        label="train",
    )
    if args.max_train_examples > 0:
        train_features = train_features[: args.max_train_examples]
        print(f"  SMOKE RUN: truncated to {len(train_features)} training features", flush=True)
    # Validation uses the PRODUCTION encoder, so an epoch metric here is directly
    # comparable to `scripts/evaluate_e3_checkpoint_0062.py` and to the shipped runtime.
    validation_examples = list(iter_jsonl(splits["validation"].path))
    validation_encoded = [
        (
            str(ex["example_id"]),
            str(ex["text"]),
            encode_for_inference(
                str(ex["text"]),
                tokenizer,
                max_length=max_length,
                segmenter=None,
                example_id=str(ex["example_id"]),
            ),
        )
        for ex in validation_examples
    ]
    print(f"  [validation] {len(validation_encoded)} examples encoded for inference", flush=True)

    # --- model ----------------------------------------------------------------
    model_config = transformers.AutoConfig.from_pretrained(
        str(model_cfg["hf_model_id"]), revision=pinned, local_files_only=True
    )
    backbone = transformers.AutoModel.from_config(model_config)

    class MentionTokenClassifier(torch.nn.Module):
        """The trained architecture: encoder + dropout + linear multi-label head."""

        def __init__(self, base_model: Any, label_count: int) -> None:
            super().__init__()
            self.base_model = base_model
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(int(base_model.config.hidden_size), label_count)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(self.dropout(output.last_hidden_state))

    model = MentionTokenClassifier(backbone, len(ENTITY_TYPE_ORDER))
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if str(payload.get("mode", "")) != CHECKPOINT_MODE:
        raise RefinementError(f"source checkpoint mode is {payload.get('mode')!r}")
    if tuple(payload.get("entity_type_order", ())) != tuple(ENTITY_TYPE_ORDER):
        raise RefinementError("source checkpoint label space does not match this repository")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    total_parameters = int(sum(p.numel() for p in model.parameters()))
    print(f"loaded source weights strict=True | parameters {total_parameters:,}", flush=True)

    decay, no_decay, head = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith("classifier"):
            head.append(parameter)
        elif any(token in name for token in ("bias", "LayerNorm.weight")):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "lr": learning_rate, "weight_decay": float(opt_cfg["weight_decay"])},
            {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0},
            {"params": head, "lr": head_lr, "weight_decay": float(opt_cfg["weight_decay"])},
        ]
    )
    per_device = int(opt_cfg["per_device_batch_size"])
    accumulation = int(opt_cfg["gradient_accumulation_steps"])
    micro_per_epoch = -(-len(train_features) // per_device)
    steps_per_epoch = max(1, micro_per_epoch // accumulation)
    total_steps = steps_per_epoch * num_epochs
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(opt_cfg["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    boundary_weight = float(args.boundary_weight)
    if boundary_weight < 1.0:
        raise RefinementError("--boundary-weight must be >= 1.0; it upweights, never downweights")

    def masked_mention_loss(logits: Any, labels: Any, label_mask: Any, boundary: Any = None) -> Any:
        """Spec §15 span focal/BCE + type loss, masked to supervised tokens.

        Identical to the original run except that ``focal_alpha`` is a parameter.
        """
        per_element = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        if str(loss_cfg.get("type", "focal")) == "focal":
            probability = torch.sigmoid(logits)
            p_t = probability * labels + (1.0 - probability) * (1.0 - labels)
            alpha_t = focal_alpha * labels + (1.0 - focal_alpha) * (1.0 - labels)
            per_element = alpha_t * (1.0 - p_t).pow(float(loss_cfg["focal_gamma"])) * per_element
        if boundary is not None and boundary_weight != 1.0:
            # Upweight the tokens that decide where a mention STOPS. The normalizer is
            # left as the supervised-token count so the loss stays on the same scale as
            # Audit 0062's: this changes the gradient's shape, not its magnitude.
            per_element = per_element * (1.0 + (boundary_weight - 1.0) * boundary)
        mask = label_mask.unsqueeze(-1)
        normalizer = torch.clamp(label_mask.sum() * len(ENTITY_TYPE_ORDER), min=1.0)
        return (per_element * mask).sum() / normalizer

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "run_id": args.run_id,
        "git_commit": git_commit(),
        "architecture_pdf_sha256": sha256_file(REPO / "docs" / "MedNorm-VI_Architecture.pdf"),
        "config_path": str(
            args.config.relative_to(REPO) if args.config.is_absolute() else args.config
        ),
        "config_sha256": config.sha256,
        "train_split_sha256": splits["train"].sha256,
        "validation_split_sha256": splits["validation"].sha256,
        "base_model_id": str(model_cfg["hf_model_id"]),
        "base_model_revision": pinned,
        "source_checkpoint": str(model_cfg["source_checkpoint"]),
        "source_checkpoint_sha256": source_sha,
        "tokenizer_identity": str(model_cfg["hf_model_id"]),
        "tokenizer_is_fast": False,
        "label_vocabulary": list(ENTITY_TYPE_ORDER),
        "seed": seed,
        "total_parameters": total_parameters,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "head_learning_rate": head_lr,
            "num_epochs": num_epochs,
            "per_device_batch_size": per_device,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": per_device * accumulation,
            "weight_decay": float(opt_cfg["weight_decay"]),
            "warmup_ratio": float(opt_cfg["warmup_ratio"]),
            "warmup_steps": int(total_steps * float(opt_cfg["warmup_ratio"])),
            "max_grad_norm": float(opt_cfg["max_grad_norm"]),
            "max_sequence_length": max_length,
            "loss_type": str(loss_cfg.get("type")),
            "focal_gamma": float(loss_cfg["focal_gamma"]),
            "focal_alpha": focal_alpha,
            "decision_threshold": threshold,
            "boundary_weight": boundary_weight,
            "source_balanced_sampling": bool(args.source_balanced),
            "mixed_precision": str(amp_dtype),
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps,
        },
        "best_metric_key": BEST_METRIC_KEY,
        "early_stopping_patience": int(eval_cfg.get("early_stopping_patience", 1)),
        "train_features": len(train_features),
        "validation_examples": len(validation_encoded),
        "smoke_truncated": bool(args.max_train_examples > 0),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": int(properties.total_memory),
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    start_epoch, global_step, best_metric, patience = 0, 0, None, 0
    latest_path = out_dir / "latest.pt"
    if args.resume and latest_path.is_file():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        if str(state.get("config_sha256", "")) != config.sha256:
            raise RefinementError("cannot resume: the checkpoint was written by another config")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        best_metric = state.get("best_metric")
        print(f"RESUMED epoch={start_epoch} step={global_step} best={best_metric}", flush=True)

    def log(record: dict[str, Any]) -> None:
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def save(destination: Path, epoch: int, step: int, best: float | None) -> str:
        torch.save(
            {
                "mode": CHECKPOINT_MODE,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": int(epoch),
                "global_step": int(step),
                "best_metric": float(best if best is not None else 0.0),
                "entity_type_order": list(ENTITY_TYPE_ORDER),
                "seed": seed,
                "pinned_model_revision": pinned,
                "config_sha256": config.sha256,
                "refinement_source_sha256": source_sha,
                "refinement_run_id": args.run_id,
            },
            destination,
        )
        return sha256_file(destination)

    rng = random.Random(seed)
    best_path = out_dir / "best.pt"
    started = time.time()
    peak_vram = 0
    model.train()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, num_epochs):
        epoch_started = time.time()
        running, micro_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        micro_index = 0
        epoch_order = source_balanced_order(train_features, rng) if args.source_balanced else None
        for batch in batches(
            train_features,
            per_device,
            pad_token_id=pad_token_id,
            shuffle_rng=rng,
            order=epoch_order,
        ):
            ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
            attention = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
            labels = torch.tensor(batch["labels"], dtype=torch.float, device=device)
            label_mask = torch.tensor(batch["label_mask"], dtype=torch.float, device=device)
            boundary = torch.tensor(batch["boundary"], dtype=torch.float, device=device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(ids, attention)
                loss = masked_mention_loss(logits, labels, label_mask, boundary)
            if not torch.isfinite(loss).item():
                raise RefinementError(f"non-finite loss at step {global_step}")
            scaled = loss / accumulation
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            running += float(loss.detach().float().cpu())
            micro_batches += 1
            micro_index += 1
            if micro_index % accumulation == 0 or micro_index == micro_per_epoch:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["max_grad_norm"]))
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 25 == 0:
                    record = {
                        "event": "train",
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": running / max(1, micro_batches),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "seconds": round(time.time() - epoch_started, 1),
                        "steps_per_epoch": steps_per_epoch,
                    }
                    log(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
                    running, micro_batches = 0.0, 0

        metrics = evaluate_exact(
            model,
            torch,
            validation_encoded,
            validation_examples,
            pad_token_id=pad_token_id,
            threshold=threshold,
            batch_size=per_device,
            amp_dtype=amp_dtype,
            device=device,
        )
        peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
        candidate = float(metrics[BEST_METRIC_KEY])
        improved = best_metric is None or candidate > float(best_metric)
        latest_sha = save(latest_path, epoch + 1, global_step, best_metric)
        best_sha = ""
        if improved:
            best_metric = candidate
            best_sha = save(best_path, epoch + 1, global_step, best_metric)
            patience = 0
        else:
            patience += 1
        record = {
            "event": "validation",
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_metric_improved": improved,
            "best_metric": best_metric,
            "latest_checkpoint_sha256": latest_sha,
            "best_checkpoint_sha256": best_sha,
            "epoch_seconds": round(time.time() - epoch_started, 1),
            "peak_vram_bytes": peak_vram,
            **metrics,
        }
        log(record)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        if patience > int(eval_cfg.get("early_stopping_patience", 1)):
            print(f"early stopping at epoch {epoch + 1}", flush=True)
            break

    final = {
        **manifest,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 1),
        "peak_vram_bytes": peak_vram,
        "peak_vram_gib": round(peak_vram / (1 << 30), 3),
        "best_metric": best_metric,
        "best_checkpoint": str(best_path) if best_path.is_file() else "",
        "best_checkpoint_sha256": sha256_file(best_path) if best_path.is_file() else "",
        "best_checkpoint_bytes": best_path.stat().st_size if best_path.is_file() else 0,
        "source_checkpoint_sha256_after_run": sha256_file(source_checkpoint),
    }
    if final["source_checkpoint_sha256_after_run"] != source_sha:
        raise RefinementError("the source checkpoint changed during the run")
    (out_dir / "experiment_manifest.json").write_text(
        json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: final[k]
                for k in (
                    "run_id",
                    "best_metric",
                    "best_checkpoint_sha256",
                    "wall_seconds",
                    "peak_vram_gib",
                )
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
