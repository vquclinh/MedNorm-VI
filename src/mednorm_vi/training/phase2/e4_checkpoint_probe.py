"""Read-only E4 checkpoint probe (Audit 0044).

Audit 0043 left one question open: what does the trained W2NER classifier
actually emit? It could not be answered because the checkpoints were absent, and
the diagnosis reported the gap honestly — but it also never contained an
executable probe. `run_collapse_diagnosis` bound `probes = ()` and only called
`require_checkpoint()` for its raise-on-absence side effect, so once the weights
arrived the probe still did not run. This module supplies the missing execution.

Everything here is **forward-only**:

* no optimizer is constructed, ever;
* no ``.backward()``, no ``.step()``, no gradient is enabled — the probe runs
  under ``torch.no_grad()`` with both modules in ``eval()``;
* no weight is written; state dicts are loaded, never saved;
* the immutable artifact is opened read-only and never written to;
* ``internal_test`` is refused by name;
* only counts, label ids, offsets and logit statistics leave a report — no
  clinical text and no tensor values.

**The base model's pretrained weights are deliberately not downloaded.** The
checkpoint carries the complete fine-tuned ``base_model`` state, so the
architecture is built from the pinned *config* alone with randomly initialized
weights, and every tensor then comes from the checkpoint. Missing and unexpected
keys are reported rather than raised on, so the report distinguishes *which* part
failed. Downloading ``pytorch_model.bin`` first and loading the checkpoint over it
would mask exactly the failure this probe exists to detect: a checkpoint that
silently does not carry the trained encoder.

Memory: checkpoints are ~4.4 GB each. They are opened one at a time with
``torch.load(..., mmap=True)`` and released before the next one is touched;
:func:`checkpoint_payload` enforces that as a context manager.
"""

from __future__ import annotations

import gc
import math
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ...mention_factory.w2ner import (
    W2NER_NONE,
    W2NERLabelVocab,
    build_relation_grid_head,
    decode_w2ner_grid,
)
from .common import sha256_file
from .e4_collapse_diagnosis import (
    CHECKPOINT_REQUIRED_KEYS,
    E4_FULL_BEST_CHECKPOINT_SHA256,
    E4_FULL_LATEST_CHECKPOINT_SHA256,
    CheckpointProbeReport,
    E4DiagnosisError,
    GovernedExample,
    assert_split_allowed,
    load_governed_examples,
)
from .e4_w2ner_training import (
    ATOMIC_PROJECTION_VERSION,
    E4_INPUT_CONTRACT_VERSION,
    E4_MODEL_ID,
    E4_PINNED_MODEL_REVISION,
    atomic_relation_head_input_dim,
    build_atomic_projection,
    build_w2ner_batch_contract_from_segmented_words,
    decode_w2ner_logits,
    prepare_phobert_word_inputs,
    project_to_atomic_word_embeddings,
)

PROBE_VERSION = "e4-checkpoint-probe-v1"

CHECKPOINT_ROLES: tuple[str, ...] = ("best", "latest")
EXPECTED_CHECKPOINT_SHA256: Mapping[str, str] = {
    "best": E4_FULL_BEST_CHECKPOINT_SHA256,
    "latest": E4_FULL_LATEST_CHECKPOINT_SHA256,
}
EXPECTED_W2NER_HEAD_KEYS: tuple[str, ...] = (
    "classifier.bias", "classifier.weight",
    "left.bias", "left.weight",
    "right.bias", "right.weight",
)

# Bounded aggregation. Logit statistics are exact for mean/min/max and use a
# deterministic stride-decimated reservoir for quantiles, so a 1.5-million-cell
# sweep never grows without limit and never depends on a random seed.
DEFAULT_SAMPLE_CAPACITY = 200_000
DEFAULT_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


class ProbeDependencyError(E4DiagnosisError):
    """A probe prerequisite is missing.

    Carries the dependency's name, the path or identifier that was looked for,
    and the exact next command — never a bare "BLOCKED".
    """

    def __init__(self, dependency: str, location: str, remedy: str) -> None:
        self.dependency = dependency
        self.location = location
        self.remedy = remedy
        super().__init__(
            f"missing probe dependency {dependency!r} at {location!r}; {remedy}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "exception_type": type(self).__name__,
            "dependency": self.dependency,
            "location": self.location,
            "next_action": self.remedy,
        }


class CheckpointRestoreError(E4DiagnosisError):
    """A checkpoint did not restore the architecture it claims to describe."""


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbeDependencies:
    """Everything the probe needs, resolved and named."""

    torch_version: str
    transformers_version: str
    model_id: str
    model_revision: str
    hidden_size: int
    relation_count: int
    tokenizer_class: str
    tokenizer_is_fast: bool
    vncorenlp_dir: str
    weights_downloaded: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "hidden_size": self.hidden_size,
            "relation_count": self.relation_count,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_is_fast": self.tokenizer_is_fast,
            "vncorenlp_dir": self.vncorenlp_dir,
            "base_model_weights_downloaded": self.weights_downloaded,
        }


def _require_module(name: str, remedy: str) -> Any:
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ProbeDependencyError(name, "python environment", remedy) from error


def resolve_probe_dependencies(
    *,
    model_id: str = E4_MODEL_ID,
    model_revision: str = E4_PINNED_MODEL_REVISION,
    cache_dir: str | Path | None = None,
    vncorenlp_dir: str | Path,
) -> ProbeDependencies:
    """Resolve every prerequisite, naming precisely what is missing if any is.

    Only the **config** and the **tokenizer** are fetched. ``pytorch_model.bin``
    is never requested: the checkpoint supplies the trained encoder weights, and
    pre-loading pretrained weights would hide a restore failure.
    """
    torch = _require_module(
        "torch", "install torch into the project environment (.venv)")
    _require_module(
        "transformers", "install transformers into the project environment (.venv)")
    _require_module(
        "py_vncorenlp",
        "install py_vncorenlp into the project environment (.venv)")

    jar = Path(vncorenlp_dir) / "VnCoreNLP-1.2.jar"
    if not jar.is_file():
        raise ProbeDependencyError(
            "VnCoreNLP-1.2.jar", str(jar),
            "run py_vncorenlp.download_model(save_dir=...) into that directory")

    from transformers import AutoConfig, AutoTokenizer
    kwargs: dict[str, Any] = {"revision": model_revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    try:
        config = AutoConfig.from_pretrained(model_id, **kwargs)
    except Exception as error:  # noqa: BLE001 - re-raised with the exact remedy
        raise ProbeDependencyError(
            f"{model_id} config.json @ {model_revision}",
            str(cache_dir or "default HuggingFace cache"),
            "make the pinned revision available locally, or allow network access "
            f"to huggingface.co/{model_id}; the probe needs config.json and the "
            "tokenizer only, never pytorch_model.bin") from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, **kwargs)
    except Exception as error:  # noqa: BLE001 - re-raised with the exact remedy
        raise ProbeDependencyError(
            f"{model_id} tokenizer @ {model_revision}",
            str(cache_dir or "default HuggingFace cache"),
            "make bpe.codes and vocab.txt available for the pinned revision") from error

    return ProbeDependencies(
        torch_version=str(torch.__version__),
        transformers_version=str(_require_module("transformers", "").__version__),
        model_id=model_id,
        model_revision=model_revision,
        hidden_size=int(config.hidden_size),
        relation_count=len(W2NERLabelVocab().labels),
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_is_fast=bool(getattr(tokenizer, "is_fast", False)),
        vncorenlp_dir=str(vncorenlp_dir),
        weights_downloaded=False,
    )


def build_probe_tokenizer(dependencies: ProbeDependencies,
                          *, cache_dir: str | Path | None = None) -> Any:
    from transformers import AutoTokenizer
    kwargs: dict[str, Any] = {"revision": dependencies.model_revision, "use_fast": False}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return AutoTokenizer.from_pretrained(dependencies.model_id, **kwargs)


def build_probe_segmenter(dependencies: ProbeDependencies) -> Callable[[str], str]:
    """Start VnCoreNLP once and return a plain ``str -> str`` segmenter.

    ``py_vncorenlp`` requires the process working directory to be its model
    directory at construction time; the original directory is restored
    immediately afterwards so nothing else in the process is affected.
    """
    import py_vncorenlp

    previous = os.getcwd()
    try:
        os.chdir(dependencies.vncorenlp_dir)
        model = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], save_dir=dependencies.vncorenlp_dir)
    finally:
        os.chdir(previous)

    def segment(text: str) -> str:
        return " ".join(model.word_segment(text))

    return segment


# ---------------------------------------------------------------------------
# B. Checkpoint payload inspection
# ---------------------------------------------------------------------------


@contextmanager
def checkpoint_payload(path: str | Path) -> Iterator[Mapping[str, Any]]:
    """Open one checkpoint read-only and guarantee it is released afterwards.

    ``mmap=True`` keeps the ~4.4 GB of tensors backed by the file instead of
    resident, and the ``finally`` block drops the reference and collects, so two
    complete payloads are never live at the same time.
    """
    torch = _require_module("torch", "install torch into the project environment")
    resolved = Path(path)
    if not resolved.is_file():
        raise ProbeDependencyError(
            resolved.name, str(resolved),
            "place the completed Colab checkpoint at that exact path")
    payload = torch.load(resolved, map_location="cpu", weights_only=False, mmap=True)
    try:
        if not isinstance(payload, dict):
            raise CheckpointRestoreError(
                f"{resolved.name}: checkpoint payload is "
                f"{type(payload).__name__}, expected a dict")
        yield payload
    finally:
        del payload
        gc.collect()


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    """Everything section B of the milestone asks for, and nothing sensitive."""

    role: str
    path: str
    sha256: str
    sha256_matches_expected: bool
    payload_type: str
    top_level_keys: tuple[str, ...]
    missing_required_keys: tuple[str, ...]
    checkpoint_schema_version: str
    e4_checkpoint_schema_version: str
    e4_input_contract_version: str
    atomic_projection_version: str
    epoch: int
    optimizer_steps: int
    backward_passes: int
    best_metric: float
    model_state_key: str
    model_state_subkeys: tuple[str, ...]
    base_model_tensor_count: int
    base_model_parameter_count: int
    w2ner_head_keys: tuple[str, ...]
    w2ner_head_shapes: Mapping[str, tuple[int, ...]]
    w2ner_head_parameter_count: int
    total_parameter_count: int
    declared_parameter_count: int
    optimizer_state_present: bool
    optimizer_state_entries: int
    scheduler_state_present: bool
    scaler_state_present: bool
    precision_mode: str
    precision_device_type: str
    autocast_dtype: str
    label_space: tuple[str, ...]
    model_revision: str
    tokenizer_revision: str
    mode: str
    expert_id: str
    internal_test_accessed: bool

    @property
    def schema_ok(self) -> bool:
        return not self.missing_required_keys and self.w2ner_head_keys == (
            EXPECTED_W2NER_HEAD_KEYS)

    @property
    def parameter_accounting_reconciles(self) -> bool:
        return self.total_parameter_count == self.declared_parameter_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_version": PROBE_VERSION,
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "sha256_matches_expected": self.sha256_matches_expected,
            "payload_type": self.payload_type,
            "top_level_keys": list(self.top_level_keys),
            "missing_required_keys": list(self.missing_required_keys),
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "e4_checkpoint_schema_version": self.e4_checkpoint_schema_version,
            "e4_input_contract_version": self.e4_input_contract_version,
            "atomic_projection_version": self.atomic_projection_version,
            "epoch": self.epoch,
            "optimizer_steps": self.optimizer_steps,
            "backward_passes": self.backward_passes,
            "best_metric": self.best_metric,
            "model_state_key": self.model_state_key,
            "model_state_subkeys": list(self.model_state_subkeys),
            "base_model_tensor_count": self.base_model_tensor_count,
            "base_model_parameter_count": self.base_model_parameter_count,
            "w2ner_head_keys": list(self.w2ner_head_keys),
            "w2ner_head_shapes": {
                key: list(value) for key, value in self.w2ner_head_shapes.items()},
            "w2ner_head_parameter_count": self.w2ner_head_parameter_count,
            "total_parameter_count": self.total_parameter_count,
            "declared_parameter_count": self.declared_parameter_count,
            "parameter_accounting_reconciles": self.parameter_accounting_reconciles,
            "optimizer_state_present": self.optimizer_state_present,
            "optimizer_state_entries": self.optimizer_state_entries,
            "scheduler_state_present": self.scheduler_state_present,
            "scaler_state_present": self.scaler_state_present,
            "precision_mode": self.precision_mode,
            "precision_device_type": self.precision_device_type,
            "autocast_dtype": self.autocast_dtype,
            "label_space": list(self.label_space),
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "mode": self.mode,
            "expert_id": self.expert_id,
            "schema_ok": self.schema_ok,
            "internal_test_accessed": self.internal_test_accessed,
            "tensor_values_read": False,
        }


def inspect_checkpoint(path: str | Path, *, role: str) -> CheckpointInspection:
    """Read one checkpoint's schema. Shapes and counts only, never values."""
    if role not in CHECKPOINT_ROLES:
        raise E4DiagnosisError(f"unknown checkpoint role {role!r}")
    resolved = Path(path)
    digest = sha256_file(resolved)
    with checkpoint_payload(resolved) as payload:
        model_state = payload.get("model_state")
        if not isinstance(model_state, Mapping):
            raise CheckpointRestoreError(
                f"{role}: model_state is {type(model_state).__name__}, expected a mapping")
        missing = tuple(
            key for key in CHECKPOINT_REQUIRED_KEYS if key not in payload)
        base = model_state.get("base_model") or {}
        head = model_state.get("w2ner_head") or {}
        head_shapes = {str(key): tuple(int(dim) for dim in value.shape)
                       for key, value in head.items()}
        base_parameters = sum(int(value.numel()) for value in base.values())
        head_parameters = sum(int(value.numel()) for value in head.values())
        optimizer_state = payload.get("optimizer_state")
        optimizer_entries = (
            len(optimizer_state.get("state", {}))
            if isinstance(optimizer_state, Mapping) else 0)
        return CheckpointInspection(
            role=role,
            path=str(resolved),
            sha256=digest,
            sha256_matches_expected=digest == EXPECTED_CHECKPOINT_SHA256.get(role, ""),
            payload_type=type(payload).__name__,
            top_level_keys=tuple(sorted(str(key) for key in payload)),
            missing_required_keys=missing,
            checkpoint_schema_version=str(payload.get("checkpoint_schema_version", "")),
            e4_checkpoint_schema_version=str(
                payload.get("e4_checkpoint_schema_version", "")),
            e4_input_contract_version=str(payload.get("e4_input_contract_version", "")),
            atomic_projection_version=str(payload.get("atomic_projection_version", "")),
            epoch=int(payload.get("epoch", -1)),
            optimizer_steps=int(payload.get("optimizer_steps", -1)),
            backward_passes=int(payload.get("backward_passes", -1)),
            best_metric=float(payload.get("best_metric", float("nan"))),
            model_state_key="model_state",
            model_state_subkeys=tuple(sorted(str(key) for key in model_state)),
            base_model_tensor_count=len(base),
            base_model_parameter_count=base_parameters,
            w2ner_head_keys=tuple(sorted(head_shapes)),
            w2ner_head_shapes=head_shapes,
            w2ner_head_parameter_count=head_parameters,
            total_parameter_count=base_parameters + head_parameters,
            declared_parameter_count=int(payload.get("parameter_count", -1)),
            optimizer_state_present=isinstance(optimizer_state, Mapping)
            and bool(optimizer_state),
            optimizer_state_entries=optimizer_entries,
            scheduler_state_present=bool(payload.get("scheduler_state")),
            scaler_state_present=bool(payload.get("scaler_state")),
            precision_mode=str(payload.get("precision_mode", "")),
            precision_device_type=str(payload.get("precision_device_type", "")),
            autocast_dtype=str(payload.get("autocast_dtype", "")),
            label_space=tuple(str(item) for item in payload.get("label_space", ())),
            model_revision=str(payload.get("model_revision", "")),
            tokenizer_revision=str(payload.get("tokenizer_revision", "")),
            mode=str(payload.get("mode", "")),
            expert_id=str(payload.get("expert_id", "")),
            internal_test_accessed=bool(payload.get("internal_test_accessed", False)),
        )


def compare_head_tensors(first: str | Path, second: str | Path) -> dict[str, Any]:
    """Do two checkpoints hold a *different* trained head?

    Loaded one at a time: the first checkpoint's head is summarized to a handful
    of scalars, released, and only then is the second opened. No tensor value is
    reported — only whether the summaries differ.
    """
    torch = _require_module("torch", "install torch into the project environment")

    def summarize(path: str | Path) -> dict[str, tuple[float, float]]:
        with checkpoint_payload(path) as payload:
            head = payload["model_state"]["w2ner_head"]
            return {
                str(key): (
                    float(torch.sum(value.double()).item()),
                    float(torch.sum(value.double() * value.double()).item()),
                )
                for key, value in head.items()
            }

    first_summary = summarize(first)
    second_summary = summarize(second)
    differing = tuple(sorted(
        key for key in first_summary
        if first_summary.get(key) != second_summary.get(key)))
    return {
        "compared_keys": sorted(first_summary),
        "differing_keys": list(differing),
        "heads_differ": bool(differing),
        "loaded_simultaneously": False,
        "tensor_values_reported": False,
    }


# ---------------------------------------------------------------------------
# C. Restoration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RestorationReport:
    """What happened when the checkpoint was loaded into a fresh architecture."""

    role: str
    strictness: str
    base_missing_keys: tuple[str, ...]
    base_unexpected_keys: tuple[str, ...]
    head_missing_keys: tuple[str, ...]
    head_unexpected_keys: tuple[str, ...]
    w2ner_head_restored: bool
    base_model_restored: bool
    instantiated_parameter_count: int
    checkpoint_parameter_count: int
    checkpoint_epoch: int
    base_weights_downloaded: bool
    initialization: str

    @property
    def ok(self) -> bool:
        return (self.base_model_restored and self.w2ner_head_restored
                and not self.base_missing_keys and not self.base_unexpected_keys
                and not self.head_missing_keys and not self.head_unexpected_keys
                and self.instantiated_parameter_count == self.checkpoint_parameter_count)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "strictness": self.strictness,
            "base_missing_keys": list(self.base_missing_keys),
            "base_unexpected_keys": list(self.base_unexpected_keys),
            "head_missing_keys": list(self.head_missing_keys),
            "head_unexpected_keys": list(self.head_unexpected_keys),
            "w2ner_head_restored": self.w2ner_head_restored,
            "base_model_restored": self.base_model_restored,
            "instantiated_parameter_count": self.instantiated_parameter_count,
            "checkpoint_parameter_count": self.checkpoint_parameter_count,
            "checkpoint_epoch": self.checkpoint_epoch,
            "base_model_weights_downloaded": self.base_weights_downloaded,
            "initialization": self.initialization,
            "restoration_ok": self.ok,
        }


def restore_e4_model(
    payload: Mapping[str, Any],
    *,
    role: str,
    dependencies: ProbeDependencies,
    cache_dir: str | Path | None = None,
) -> tuple[Any, Any, RestorationReport]:
    """Build the pinned architecture and load this checkpoint into it, strictly.

    The encoder is instantiated from ``config.json`` with **no pretrained
    weights**, so every encoder tensor must come from the checkpoint. A missing
    key is then unambiguous evidence that the checkpoint does not carry the
    trained encoder, instead of being silently satisfied by a pretrained value.

    The expected PhoBERT MLM-head keys stay distinguishable from missing trained
    W2NER parameters because ``AutoModel`` builds the bare encoder, which has no
    MLM head at all: nothing about ``lm_head.*`` can appear here.
    """
    torch = _require_module("torch", "install torch into the project environment")
    from transformers import AutoConfig, AutoModel

    kwargs: dict[str, Any] = {"revision": dependencies.model_revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    config = AutoConfig.from_pretrained(dependencies.model_id, **kwargs)
    base_model = AutoModel.from_config(config)
    # build_relation_grid_head defers its torch import and is typed as returning
    # ``object`` so the module stays importable without torch; it is an nn.Module.
    head = cast(Any, build_relation_grid_head(
        atomic_relation_head_input_dim(int(config.hidden_size)),
        dependencies.relation_count))

    model_state = payload["model_state"]
    base_result = base_model.load_state_dict(model_state["base_model"], strict=False)
    head_result = head.load_state_dict(model_state["w2ner_head"], strict=False)

    instantiated = (sum(p.numel() for p in base_model.parameters())
                    + sum(p.numel() for p in head.parameters()))
    checkpoint_parameters = (
        sum(int(v.numel()) for v in model_state["base_model"].values())
        + sum(int(v.numel()) for v in model_state["w2ner_head"].values()))

    base_model.eval()
    head.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    torch.set_grad_enabled(False)

    report = RestorationReport(
        role=role,
        strictness="strict=False, then every missing/unexpected key reported",
        base_missing_keys=tuple(sorted(str(k) for k in base_result.missing_keys)),
        base_unexpected_keys=tuple(sorted(str(k) for k in base_result.unexpected_keys)),
        head_missing_keys=tuple(sorted(str(k) for k in head_result.missing_keys)),
        head_unexpected_keys=tuple(sorted(str(k) for k in head_result.unexpected_keys)),
        w2ner_head_restored=not head_result.missing_keys and not head_result.unexpected_keys,
        base_model_restored=not base_result.missing_keys and not base_result.unexpected_keys,
        instantiated_parameter_count=int(instantiated),
        checkpoint_parameter_count=int(checkpoint_parameters),
        checkpoint_epoch=int(payload.get("epoch", -1)),
        base_weights_downloaded=False,
        initialization="architecture from config.json, all weights from the checkpoint",
    )
    if not report.w2ner_head_restored:
        raise CheckpointRestoreError(
            f"{role}: the W2NER head did not restore "
            f"(missing={report.head_missing_keys}, "
            f"unexpected={report.head_unexpected_keys})")
    return base_model, head, report


# ---------------------------------------------------------------------------
# Bounded logit statistics
# ---------------------------------------------------------------------------


@dataclass
class BoundedSample:
    """Deterministic, capacity-bounded reservoir for quantiles.

    Exact ``count``/``mean``/``min``/``max`` are maintained over every observed
    value. Quantiles come from a stride-decimated subsample: values are kept
    until ``capacity``, then the reservoir is halved and the stride doubled, so
    memory is capped and the result depends only on arrival order — never on a
    random seed.
    """

    capacity: int = DEFAULT_SAMPLE_CAPACITY
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    stride: int = 1
    _kept: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        if (self.count - 1) % self.stride:
            return
        self._kept.append(value)
        if len(self._kept) > self.capacity:
            self._kept = self._kept[::2]
            self.stride *= 2

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    def quantiles(
        self, points: Sequence[float] = DEFAULT_QUANTILES,
    ) -> dict[str, float]:
        if not self._kept:
            return {}
        ordered = sorted(self._kept)
        result: dict[str, float] = {}
        for point in points:
            index = min(len(ordered) - 1, max(0, int(round(point * (len(ordered) - 1)))))
            result[f"p{int(point * 100):02d}"] = ordered[index]
        return result

    def as_dict(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0}
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.minimum,
            "max": self.maximum,
            "sampled_for_quantiles": len(self._kept),
            "sample_stride": self.stride,
            **self.quantiles(),
        }


# ---------------------------------------------------------------------------
# D. Read-only governed validation probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridLogitReport:
    """Cell-level evidence: what the classifier emits and where it is wrong."""

    role: str
    grid_cells: int
    predicted_labels_by_class: Mapping[str, int]
    none_predictions: int
    non_none_predictions: int
    gold_positive_cells: int
    gold_positive_predicted_as_none: int
    gold_positive_predicted_correct_class: int
    gold_positive_predicted_labels: Mapping[str, int]
    none_logits: Mapping[str, Any]
    strongest_non_none_logits: Mapping[str, Any]
    non_none_margin_over_none: Mapping[str, Any]
    gold_positive_margin_over_none: Mapping[str, Any]
    decoder_input_thw_relations: int
    decoder_input_nnw_relations: int
    decoder_output_mentions: int

    @property
    def gold_positive_background_rate(self) -> float:
        if not self.gold_positive_cells:
            return 0.0
        return self.gold_positive_predicted_as_none / self.gold_positive_cells

    @property
    def gold_positive_correct_class_rate(self) -> float:
        if not self.gold_positive_cells:
            return 0.0
        return self.gold_positive_predicted_correct_class / self.gold_positive_cells

    @property
    def non_none_prediction_rate(self) -> float:
        return self.non_none_predictions / self.grid_cells if self.grid_cells else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "grid_cells": self.grid_cells,
            "predicted_labels_by_class": dict(self.predicted_labels_by_class),
            "none_predictions": self.none_predictions,
            "non_none_predictions": self.non_none_predictions,
            "non_none_prediction_rate": self.non_none_prediction_rate,
            "gold_positive_cells": self.gold_positive_cells,
            "gold_positive_predicted_as_none": self.gold_positive_predicted_as_none,
            "gold_positive_background_rate": self.gold_positive_background_rate,
            "gold_positive_predicted_correct_class": (
                self.gold_positive_predicted_correct_class),
            "gold_positive_correct_class_rate": self.gold_positive_correct_class_rate,
            "gold_positive_predicted_labels": dict(self.gold_positive_predicted_labels),
            "none_logits": dict(self.none_logits),
            "strongest_non_none_logits": dict(self.strongest_non_none_logits),
            "non_none_margin_over_none": dict(self.non_none_margin_over_none),
            "gold_positive_margin_over_none": dict(self.gold_positive_margin_over_none),
            "decoder_input_thw_relations": self.decoder_input_thw_relations,
            "decoder_input_nnw_relations": self.decoder_input_nnw_relations,
            "decoder_output_mentions": self.decoder_output_mentions,
            "internal_test_accessed": False,
        }


def _build_contract(
    example: GovernedExample,
    *,
    segmenter: Callable[[str], str],
    max_words: int,
    vocab: W2NERLabelVocab,
) -> Any:
    from ..phobert_alignment import (
        map_segmented_words,
        resolve_segmented_text,
        segmented_text_to_words,
    )
    segmented_text, _source = resolve_segmented_text(example.text, segmenter)
    model_words = map_segmented_words(
        example.text, segmented_text_to_words(segmented_text))
    return build_w2ner_batch_contract_from_segmented_words(
        example.document_id, example.text, example.entities, model_words,
        max_words=max_words, vocab=vocab)


def probe_checkpoint_on_validation(
    *,
    role: str,
    checkpoint_path: str | Path,
    validation_path: str | Path,
    dependencies: ProbeDependencies,
    tokenizer: Any,
    segmenter: Callable[[str], str],
    cache_dir: str | Path | None = None,
    split: str = "validation",
    max_words: int = 256,
    max_model_tokens: int = 256,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[CheckpointProbeReport, GridLogitReport, RestorationReport]:
    """Forward-only inference over governed validation for one checkpoint.

    ``torch.no_grad()`` wraps everything, both modules are in ``eval()``, every
    parameter has ``requires_grad`` cleared, and no optimizer or backward call
    exists anywhere in this function.
    """
    assert_split_allowed(split)
    torch = _require_module("torch", "install torch into the project environment")
    vocab = W2NERLabelVocab()
    labels = vocab.labels
    none_id = vocab.none_id

    with checkpoint_payload(checkpoint_path) as payload:
        base_model, head, restoration = restore_e4_model(
            payload, role=role, dependencies=dependencies, cache_dir=cache_dir)

    predicted_total = gold_total = true_positives = 0
    predictions_by_type: dict[str, int] = {}
    predicted_labels: dict[str, int] = dict.fromkeys(labels, 0)
    gold_cell_labels: dict[str, int] = dict.fromkeys(labels, 0)
    grid_cells = gold_positive_cells = 0
    gold_positive_as_none = gold_positive_correct = 0
    thw_relations = nnw_relations = 0
    decoder_output = 0

    none_sample = BoundedSample()
    best_non_none_sample = BoundedSample()
    margin_sample = BoundedSample()
    gold_margin_sample = BoundedSample()

    examples = list(load_governed_examples(validation_path, split=split, limit=limit))
    total = len(examples)

    with torch.no_grad():
        for index, example in enumerate(examples):
            contract = _build_contract(
                example, segmenter=segmenter, max_words=max_words, vocab=vocab)
            encoding = prepare_phobert_word_inputs(
                tokenizer, contract.segmented_words, max_length=max_model_tokens)
            projection = build_atomic_projection(
                contract.grid.original_text, contract.segmented_words, encoding,
                atomic_words=contract.atomic_words)
            model_inputs = {
                key: torch.tensor([value], dtype=torch.long)
                for key, value in encoding.model_inputs.items()
            }
            outputs = base_model(**model_inputs)
            embeddings = project_to_atomic_word_embeddings(
                outputs.last_hidden_state[0], projection)
            pair_mask = torch.tensor([contract.grid.pair_mask], dtype=torch.bool)
            logits = head(embeddings, pair_mask)[0]

            size = len(contract.grid.words)
            window = logits[:size, :size, :]
            argmax = window.argmax(dim=-1)
            none_logit = window[:, :, none_id]
            without_none = torch.cat(
                (window[:, :, :none_id], window[:, :, none_id + 1:]), dim=-1)
            best_non_none = without_none.max(dim=-1).values
            margin = best_non_none - none_logit

            grid_cells += size * size
            counts = torch.bincount(argmax.reshape(-1), minlength=len(labels))
            for label_id, count in enumerate(counts.tolist()):
                predicted_labels[labels[label_id]] += int(count)

            for value in none_logit.reshape(-1).tolist():
                none_sample.observe(float(value))
            for value in best_non_none.reshape(-1).tolist():
                best_non_none_sample.observe(float(value))
            for value in margin.reshape(-1).tolist():
                margin_sample.observe(float(value))

            target = contract.grid.labels
            argmax_rows = argmax.tolist()
            for row_index in range(size):
                target_row = target[row_index]
                predicted_row = argmax_rows[row_index]
                for column_index in range(size):
                    gold_label = target_row[column_index]
                    if gold_label == none_id:
                        continue
                    gold_positive_cells += 1
                    predicted_label = predicted_row[column_index]
                    gold_cell_labels[labels[predicted_label]] += 1
                    if predicted_label == none_id:
                        gold_positive_as_none += 1
                    if predicted_label == gold_label:
                        gold_positive_correct += 1
                    gold_margin_sample.observe(
                        float(margin[row_index][column_index].item()))

            thw_relations += int(sum(
                1 for row in argmax_rows for value in row
                if vocab.thw_type(value) is not None))
            nnw_relations += int(sum(
                1 for row in argmax_rows for value in row if value == vocab.nnw_id))

            predicted = set(decode_w2ner_logits(contract, logits.tolist()))
            gold = {(span.start, span.end, span.entity_type)
                    for span in decode_w2ner_grid(contract.grid)}
            decoder_output += len(predicted)
            predicted_total += len(predicted)
            gold_total += len(gold)
            true_positives += len(predicted & gold)
            for _start, _end, entity_type in predicted:
                predictions_by_type[entity_type] = (
                    predictions_by_type.get(entity_type, 0) + 1)

            del (contract, encoding, projection, model_inputs, outputs, embeddings,
                 pair_mask, logits, window, argmax, none_logit, without_none,
                 best_non_none, margin, predicted, gold)
            if progress is not None:
                progress(index + 1, total)

    del base_model, head
    gc.collect()

    precision = true_positives / predicted_total if predicted_total else 0.0
    recall = true_positives / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    logit_report = GridLogitReport(
        role=role,
        grid_cells=grid_cells,
        predicted_labels_by_class=dict(predicted_labels),
        none_predictions=predicted_labels[W2NER_NONE],
        non_none_predictions=grid_cells - predicted_labels[W2NER_NONE],
        gold_positive_cells=gold_positive_cells,
        gold_positive_predicted_as_none=gold_positive_as_none,
        gold_positive_predicted_correct_class=gold_positive_correct,
        gold_positive_predicted_labels=dict(gold_cell_labels),
        none_logits=none_sample.as_dict(),
        strongest_non_none_logits=best_non_none_sample.as_dict(),
        non_none_margin_over_none=margin_sample.as_dict(),
        gold_positive_margin_over_none=gold_margin_sample.as_dict(),
        decoder_input_thw_relations=thw_relations,
        decoder_input_nnw_relations=nnw_relations,
        decoder_output_mentions=decoder_output,
    )
    probe_report = CheckpointProbeReport(
        role=role,
        checkpoint_sha256=sha256_file(Path(checkpoint_path)),
        epoch=restoration.checkpoint_epoch,
        predicted_mention_total=predicted_total,
        gold_mention_total=gold_total,
        true_positives=true_positives,
        exact_precision=precision,
        exact_recall=recall,
        exact_f1=f1,
        predictions_by_entity_type=dict(sorted(predictions_by_type.items())),
        predicted_labels_by_class=dict(predicted_labels),
        background_label_count=predicted_labels[W2NER_NONE],
        non_background_label_count=grid_cells - predicted_labels[W2NER_NONE],
        background_logit_quantiles=none_sample.quantiles(),
        strongest_non_background_logit_quantiles=best_non_none_sample.quantiles(),
        gold_positive_cell_predicted_labels=dict(gold_cell_labels),
        gold_positive_cell_background_rate=logit_report.gold_positive_background_rate,
        decoder_input_positive_relations=thw_relations,
        decoder_output_mention_count=decoder_output,
        w2ner_head_keys_restored=EXPECTED_W2NER_HEAD_KEYS,
    )
    return probe_report, logit_report, restoration


def classify_probe_outcome(
    probe: CheckpointProbeReport,
    logits: GridLogitReport,
    restoration: RestorationReport,
) -> str:
    """Name which of the four candidate mechanisms the evidence supports.

    The decisive quantity is the **THW** count, not the non-background count. A
    mention exists only where the grid carries a THW relation; an isolated NNW
    cell is a path edge with no head to anchor it, so a grid of pure NONE plus a
    stray NNW yields exactly as many mentions as a grid of pure NONE — zero. Any
    classifier that treated "some non-background cell exists" as evidence against
    background collapse would misread precisely this case.
    """
    if not restoration.ok:
        return "checkpoint_does_not_restore_the_trained_head"
    if logits.decoder_input_thw_relations == 0:
        return "trained_head_emits_no_entity_relation_only_background"
    if logits.decoder_output_mentions == 0:
        return "head_emits_positive_relations_but_the_decoder_loses_them"
    if probe.predicted_mention_total > 0:
        return "head_and_decoder_both_produce_mentions"
    return "another_evidenced_failure"


# Where the VnCoreNLP segmenter models live for a local probe. Overridable via
# MEDNORM_VNCORENLP_DIR; the default is the repository's existing model cache.
DEFAULT_VNCORENLP_DIR = str(Path.home() / ".cache" / "mednorm-vi" / "vncorenlp")


def run_default_checkpoint_probe(
    artifact_dir: str | Path,
    split_paths: Mapping[str, Any],
    *,
    limit: int | None = None,
) -> tuple[tuple[CheckpointProbeReport, ...], tuple[Mapping[str, Any], ...]]:
    """Probe both checkpoints for :func:`run_collapse_diagnosis`.

    Checkpoints are handled strictly one at a time — inspect, restore, sweep
    validation, release — before the next role is opened.
    """
    validation_path = split_paths.get("validation")
    if validation_path is None:
        raise ProbeDependencyError(
            "governed validation split", "split_paths['validation']",
            "pass --split-root pointing at the governed splits directory")

    vncorenlp_dir = os.environ.get(
        "MEDNORM_VNCORENLP_DIR", DEFAULT_VNCORENLP_DIR)
    dependencies = resolve_probe_dependencies(vncorenlp_dir=vncorenlp_dir)
    tokenizer = build_probe_tokenizer(dependencies)
    segmenter = build_probe_segmenter(dependencies)

    probes: list[CheckpointProbeReport] = []
    inspections: list[Mapping[str, Any]] = []
    for role in CHECKPOINT_ROLES:
        path = Path(artifact_dir) / "checkpoints" / f"{role}.pt"
        inspection = inspect_checkpoint(path, role=role)
        probe, logit_report, restoration = probe_checkpoint_on_validation(
            role=role, checkpoint_path=path, validation_path=validation_path,
            dependencies=dependencies, tokenizer=tokenizer, segmenter=segmenter,
            limit=limit)
        probes.append(probe)
        inspections.append({
            **inspection.as_dict(),
            "restoration": restoration.as_dict(),
            "grid_logits": logit_report.as_dict(),
            "outcome": classify_probe_outcome(probe, logit_report, restoration),
            # resolve_verdict() reads this key to decide CHECKPOINT_RESTORE_FAILURE.
            "w2ner_head_restored": restoration.w2ner_head_restored,
        })
        gc.collect()
    return tuple(probes), tuple(inspections)


__all__ = [
    "CHECKPOINT_ROLES",
    "DEFAULT_VNCORENLP_DIR",
    "run_default_checkpoint_probe",
    "DEFAULT_QUANTILES",
    "DEFAULT_SAMPLE_CAPACITY",
    "EXPECTED_CHECKPOINT_SHA256",
    "EXPECTED_W2NER_HEAD_KEYS",
    "PROBE_VERSION",
    "ATOMIC_PROJECTION_VERSION",
    "E4_INPUT_CONTRACT_VERSION",
    "BoundedSample",
    "CheckpointInspection",
    "CheckpointRestoreError",
    "GridLogitReport",
    "ProbeDependencies",
    "ProbeDependencyError",
    "RestorationReport",
    "build_probe_segmenter",
    "build_probe_tokenizer",
    "checkpoint_payload",
    "classify_probe_outcome",
    "compare_head_tensors",
    "inspect_checkpoint",
    "probe_checkpoint_on_validation",
    "resolve_probe_dependencies",
    "restore_e4_model",
]
