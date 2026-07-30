"""Forward-only execution of the validated S1 ViHealthBERT span/type checkpoint.

This is the tracked version of the recipe Audit 0033 executed by hand::

    checkpoint validation
    -> optional VnCoreNLP RDRSegmenter when explicitly configured
    -> slow tokenizer at the pinned revision
    -> AutoConfig + AutoModel.from_config
    -> strict checkpoint state-dict load
    -> model.eval() + torch.no_grad()
    -> character-offset decoding

**Execution boundary.** Nothing here trains. There is no ``backward``, no
optimizer, no scheduler, no ``requires_grad`` parameter, and no write of any kind
to the checkpoint. Base-model weights are never downloaded: the architecture is
rebuilt from the cached ``config.json`` and the checkpoint's own complete state
dict, loaded with ``strict=True``. The checkpoint digest and mtime are captured
before and after every run so a mutation cannot pass unnoticed.

Torch and Transformers are imported lazily inside the functions that need them,
so importing this module costs nothing and the rest of the package stays free of
heavy dependencies.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...training.phobert_alignment import (
    align_subtokens,
    map_segmented_words,
    resolve_segmented_text,
    segmented_text_to_words,
)
from ...training.s1_mention_smoke import ENTITY_TYPE_ORDER
from .checkpoint_loading import assert_load_is_permitted, describe_policy
from .decoding import (
    DECODING_POLICY,
    EXPERT_ID,
    NeuralSpan,
    decode_type_runs,
    validate_decoded_spans,
)


class CheckpointIntegrityError(RuntimeError):
    """Raised when the checkpoint does not match the accepted digest or moved."""


@dataclass(frozen=True, slots=True)
class VnCoreNLPCapability:
    """Lightweight readiness result for an explicitly configured VnCoreNLP path."""

    ready: bool
    reason_code: str | None = None
    detail: str | None = None

    def blocker(self) -> str:
        if self.ready:
            return ""
        reason = self.reason_code or "unknown"
        detail = f":{self.detail}" if self.detail else ""
        return f"e3_vncorenlp_not_ready:{reason}{detail}"


def _module_spec_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _import_vncorenlp_dependency(module_name: str) -> VnCoreNLPCapability:
    missing_reason = f"missing_{module_name.replace('.', '_')}"
    if not _module_spec_available(module_name):
        return VnCoreNLPCapability(False, missing_reason, module_name)
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or module_name
        if missing_name == "jnius_config":
            return VnCoreNLPCapability(False, "missing_jnius_config", missing_name)
        if missing_name == "py_vncorenlp":
            return VnCoreNLPCapability(False, "missing_py_vncorenlp", missing_name)
        return VnCoreNLPCapability(
            False,
            "runtime_import_failed",
            f"{module_name}:{type(exc).__name__}:{missing_name}",
        )
    except ImportError as exc:
        return VnCoreNLPCapability(
            False,
            "runtime_import_failed",
            f"{module_name}:{type(exc).__name__}:{exc}",
        )
    except Exception as exc:  # noqa: BLE001 - typed dependency probe preserves the cause
        return VnCoreNLPCapability(
            False,
            "runtime_import_failed",
            f"{module_name}:{type(exc).__name__}:{exc}",
        )
    return VnCoreNLPCapability(True)


def probe_vncorenlp_capability(vncorenlp_dir: str | Path) -> VnCoreNLPCapability:
    """Check the optional VnCoreNLP integration without loading E3 or a tokenizer."""
    directory = Path(vncorenlp_dir)
    if not str(vncorenlp_dir).strip():
        return VnCoreNLPCapability(False, "missing_path", "")
    if not directory.exists():
        return VnCoreNLPCapability(False, "missing_path", str(directory))
    if not directory.is_dir():
        return VnCoreNLPCapability(False, "not_directory", str(directory))

    jars = tuple(path for path in directory.glob("*.jar") if path.is_file())
    if not jars:
        return VnCoreNLPCapability(False, "missing_jar", str(directory))
    if not any(path.stat().st_size > 0 for path in jars):
        return VnCoreNLPCapability(False, "empty_jar", str(directory))

    if shutil.which("java") is None:
        return VnCoreNLPCapability(False, "missing_java_runtime", "java")

    jnius = _import_vncorenlp_dependency("jnius_config")
    if not jnius.ready:
        return jnius
    py_vncorenlp = _import_vncorenlp_dependency("py_vncorenlp")
    if not py_vncorenlp.ready:
        return py_vncorenlp
    return VnCoreNLPCapability(True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointFingerprint:
    """Digest + filesystem identity of the checkpoint, captured at a point in time."""

    path: str
    sha256: str
    size_bytes: int
    mtime_ns: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }

    def assert_unchanged(self, other: CheckpointFingerprint) -> None:
        """Fail loudly if anything about the checkpoint file moved."""
        if (self.sha256, self.size_bytes, self.mtime_ns) != (
            other.sha256,
            other.size_bytes,
            other.mtime_ns,
        ):
            raise CheckpointIntegrityError(
                f"checkpoint changed during execution: {self.as_dict()} != {other.as_dict()}"
            )


def fingerprint_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str = "",
) -> CheckpointFingerprint:
    """Hash the checkpoint and record its filesystem identity.

    ``expected_sha256`` is compared when supplied; a mismatch raises before the
    bytes are ever loaded, so an unexpected checkpoint can never be executed.
    """
    target = Path(path)
    if not target.is_file():
        raise CheckpointIntegrityError(f"checkpoint not found: {target}")
    stat = target.stat()
    digest = sha256_file(target)
    if expected_sha256 and digest != expected_sha256.strip().lower():
        raise CheckpointIntegrityError(
            f"checkpoint digest mismatch for {target}: expected "
            f"{expected_sha256.strip().lower()}, found {digest}"
        )
    return CheckpointFingerprint(str(target), digest, stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True, slots=True)
class NeuralExpertConfig:
    """Everything the E3 forward pass needs. All values are tracked or supplied."""

    checkpoint_path: str
    expected_checkpoint_sha256: str
    pinned_model_revision: str
    hf_model_id: str = "demdecuong/vihealthbert-base-word"
    max_sequence_length: int = 256
    decision_threshold: float = 0.5
    batch_size: int = 8
    model_cache_dir: str = ""
    vncorenlp_dir: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
            "pinned_model_revision": self.pinned_model_revision,
            "hf_model_id": self.hf_model_id,
            "max_sequence_length": self.max_sequence_length,
            "decision_threshold": self.decision_threshold,
            "entity_type_order": list(ENTITY_TYPE_ORDER),
            "decoding_policy": DECODING_POLICY,
            "expert_id": EXPERT_ID,
        }


@dataclass(frozen=True, slots=True)
class EncodedExample:
    """One inference-only encoding: ids plus each subtoken's original span."""

    example_id: str
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    truncated: bool = False


def encode_for_inference(
    original_text: str,
    tokenizer: Any,
    *,
    max_length: int,
    segmenter: Callable[[str], str] | None,
    example_id: str = "",
) -> EncodedExample:
    """Encode one example for a forward pass — offsets only, no labels.

    Uses exactly the segmentation and alignment contracts training used
    (:mod:`mednorm_vi.training.phobert_alignment`), so the subtoken sequence the
    model sees at inference is the sequence it was trained on. Labels are
    deliberately not built: nothing here supervises anything.
    """
    segmented, _source = resolve_segmented_text(original_text, segmenter)
    words = map_segmented_words(original_text, segmented_text_to_words(segmented))
    aligned = align_subtokens(
        words,
        tokenizer,
        max_length=max_length,
        cls_token_id=getattr(tokenizer, "cls_token_id", None),
        sep_token_id=getattr(tokenizer, "sep_token_id", None),
    )
    return EncodedExample(
        example_id=example_id,
        input_ids=tuple(piece.token_id for piece in aligned.subtokens),
        offsets=tuple(
            (0, 0) if piece.is_special else (piece.original_start, piece.original_end)
            for piece in aligned.subtokens
        ),
        truncated=aligned.truncated,
    )


@dataclass
class _RuntimeReport:
    examples: int = 0
    truncated_examples: int = 0
    decoded_spans: int = 0
    backend: dict[str, Any] = field(default_factory=dict)


class S1NeuralExpert:
    """E3: the trained ViHealthBERT span/type expert, forward-only.

    Construct through :func:`load_expert`. ``predict_spans`` returns decoded,
    character-exact :class:`~.decoding.NeuralSpan` objects — proposals, never
    final organizer entities (spec §6).
    """

    def __init__(
        self,
        config: NeuralExpertConfig,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        segmenter: Callable[[str], str] | None,
        fingerprint: CheckpointFingerprint,
        backend: Mapping[str, Any],
    ) -> None:
        self.config = config
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch_module
        self._segmenter = segmenter
        self.initial_fingerprint = fingerprint
        self.report = _RuntimeReport(backend=dict(backend))

    # -- integrity -------------------------------------------------------------
    def verify_checkpoint_unchanged(self) -> CheckpointFingerprint:
        """Re-hash the checkpoint and assert nothing about it moved."""
        current = fingerprint_checkpoint(self.config.checkpoint_path)
        self.initial_fingerprint.assert_unchanged(current)
        return current

    # -- inference -------------------------------------------------------------
    def predict_spans(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> dict[str, tuple[NeuralSpan, ...]]:
        """Decode character-exact mentions for each example. Forward passes only."""
        torch = self._torch
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 1
        threshold = float(self.config.decision_threshold)

        encoded: list[tuple[str, str, EncodedExample]] = []
        for example in examples:
            example_id = str(example.get("example_id", ""))
            text = str(example.get("text", ""))
            encoding = encode_for_inference(
                text,
                self._tokenizer,
                max_length=self.config.max_sequence_length,
                segmenter=self._segmenter,
                example_id=example_id,
            )
            encoded.append((example_id, text, encoding))
            if encoding.truncated:
                self.report.truncated_examples += 1

        out: dict[str, tuple[NeuralSpan, ...]] = {}
        with torch.no_grad():
            for start in range(0, len(encoded), self.config.batch_size):
                batch = encoded[start : start + self.config.batch_size]
                width = max(len(item[2].input_ids) for item in batch)
                input_ids = [
                    list(item[2].input_ids) + [int(pad_token_id)] * (width - len(item[2].input_ids))
                    for item in batch
                ]
                attention = [
                    [1] * len(item[2].input_ids) + [0] * (width - len(item[2].input_ids))
                    for item in batch
                ]
                logits = self._model(
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(attention, dtype=torch.long),
                )
                probabilities = torch.sigmoid(logits.float())
                positives = (probabilities > threshold).int().tolist()
                probability_rows = probabilities.tolist()
                for index, (example_id, text, encoding) in enumerate(batch):
                    length = len(encoding.input_ids)
                    spans = decode_type_runs(
                        text,
                        encoding.offsets,
                        positives[index][:length],
                        ENTITY_TYPE_ORDER,
                        probabilities=probability_rows[index][:length],
                    )
                    validate_decoded_spans(spans, text)
                    out[example_id] = spans
                    self.report.examples += 1
                    self.report.decoded_spans += len(spans)
        return out


def build_segmenter(vncorenlp_dir: str | Path) -> Callable[[str], str]:
    """Production RDRSegmenter callable for an explicitly configured path.

    The shipped default E3 profile passes ``segmenter=None`` and uses the
    pre-segmented-source path. Once a caller configures ``e3_vncorenlp_dir``,
    however, VnCoreNLP is mandatory for that run: readiness and runtime share
    :func:`probe_vncorenlp_capability`, and this constructor never falls back to
    ``None``.
    """
    directory = Path(vncorenlp_dir)
    capability = probe_vncorenlp_capability(directory)
    if not capability.ready:
        raise RuntimeError(capability.blocker())
    py_vncorenlp = importlib.import_module("py_vncorenlp")
    # py_vncorenlp chdirs into the JVM working directory on construction.
    cwd = Path.cwd()
    try:
        try:
            rdr = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(directory))
        except Exception as exc:  # noqa: BLE001 - runtime path names the failing capability
            raise RuntimeError(
                f"e3_vncorenlp_not_ready:runtime_construction_failed:{type(exc).__name__}:{exc}"
            ) from exc
    finally:
        os.chdir(cwd)

    def segment(text: str) -> str:
        segments = rdr.word_segment(text)
        if not segments:
            raise RuntimeError("RDRSegmenter returned no segments")
        return " ".join(str(segment) for segment in segments)

    return segment


def load_expert(
    config: NeuralExpertConfig,
    *,
    segmenter: Callable[[str], str] | None = None,
) -> S1NeuralExpert:
    """Validate the checkpoint, rebuild the architecture, and load it strictly.

    The base weights are **never** downloaded. ``AutoConfig`` supplies the
    architecture from the local cache and the checkpoint's complete state dict
    supplies every parameter, so the result is bit-identical to the validated run.
    """
    fingerprint = fingerprint_checkpoint(
        config.checkpoint_path, expected_sha256=config.expected_checkpoint_sha256
    )

    if config.model_cache_dir:
        os.environ.setdefault("HF_HOME", config.model_cache_dir)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.hf_model_id,
        revision=config.pinned_model_revision,
        cache_dir=config.model_cache_dir or None,
        use_fast=False,
        local_files_only=True,
    )
    if bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError(
            "expected the slow PhobertTokenizer; a fast tokenizer would change "
            "the alignment contract the checkpoint was trained under"
        )

    # How this file may be deserialized is a policy decision, not a call-site habit
    # (Audit 0056a). The digest was computed above, before any bytes were
    # interpreted; `assert_load_is_permitted` is what turns "we happen to hash first"
    # into "an unverified checkpoint cannot reach weights_only=False". Only E3's one
    # validated legacy artifact is admitted to the pickle path; anything else must be
    # safetensors or weights_only-loadable.
    policy = assert_load_is_permitted(
        config.checkpoint_path,
        verified_sha256=fingerprint.sha256,
        expected_sha256=config.expected_checkpoint_sha256,
    )
    payload = torch.load(
        config.checkpoint_path, map_location="cpu", weights_only=policy.weights_only
    )
    if str(payload.get("mode", "")) != "FULL_TRAINING":
        raise CheckpointIntegrityError(
            f"checkpoint mode is {payload.get('mode')!r}, not FULL_TRAINING"
        )
    if tuple(payload.get("entity_type_order", ())) != tuple(ENTITY_TYPE_ORDER):
        raise CheckpointIntegrityError("checkpoint label space does not match ENTITY_TYPE_ORDER")
    revision = str(payload.get("pinned_model_revision", ""))
    if revision and revision != config.pinned_model_revision:
        raise CheckpointIntegrityError(
            f"checkpoint was trained at revision {revision}, not {config.pinned_model_revision}"
        )

    model_config = transformers.AutoConfig.from_pretrained(
        config.hf_model_id,
        revision=config.pinned_model_revision,
        cache_dir=config.model_cache_dir or None,
        local_files_only=True,
    )
    backbone = transformers.AutoModel.from_config(model_config)

    class MentionTokenClassifier(torch.nn.Module):  # type: ignore[misc, name-defined]
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
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    backend = {
        "backbone_source": "config_only_then_strict_state_dict",
        "base_weights_downloaded": False,
        "strict_state_dict_load": True,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "optimizer_constructed": False,
        "scheduler_constructed": False,
        "backward_called": False,
        "decoding_policy": DECODING_POLICY,
        # Which deserialization policy admitted this file, so a run manifest records
        # that the legacy pickle allowance was used and for which artifact.
        **describe_policy(policy),
    }
    fingerprint.assert_unchanged(fingerprint_checkpoint(config.checkpoint_path))
    return S1NeuralExpert(config, model, tokenizer, torch, segmenter, fingerprint, backend)


def iter_examples(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream governed JSONL examples without loading the whole split."""
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield dict(json.loads(line))


__all__ = [
    "CheckpointFingerprint",
    "CheckpointIntegrityError",
    "EncodedExample",
    "NeuralExpertConfig",
    "S1NeuralExpert",
    "VnCoreNLPCapability",
    "build_segmenter",
    "encode_for_inference",
    "fingerprint_checkpoint",
    "iter_examples",
    "load_expert",
    "probe_vncorenlp_capability",
    "sha256_file",
]
