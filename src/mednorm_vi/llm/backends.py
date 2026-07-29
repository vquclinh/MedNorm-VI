"""Local-only pretrained inference backends (spec §12, §19, Appendix A).

Appendix A requires that "every model/tokenizer/KB/index uses a local path;
inference requires no external API". These backends are how that is enforced
rather than promised:

* **the model identity is a local directory**, not a hub id. A backend built on a
  path that does not exist raises :class:`LocalCheckpointMissing` at load time and
  never reaches a resolver that could download it;
* **``local_files_only=True`` is passed unconditionally** and cannot be overridden
  by a caller, so an unexpectedly reachable network can never turn a missing
  checkpoint into a silent download;
* **the heavy dependency is imported inside ``load()``**. Importing this module
  costs nothing and requires neither ``torch``, ``transformers``, ``gliner`` nor a
  GPU, which is what lets contract tests and architecture inspection run on a
  laptop with no model assets present.

Nothing here trains: no optimizer, no scheduler, no backward pass. Every forward
runs under ``torch.no_grad()`` with the module in ``eval()``.

Generation is greedy (``do_sample=False``), so a backend is deterministic by
construction — a sampled adjudication would make an ablation unreproducible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

BACKEND_CONTRACT_VERSION = "local-backend-v1"

# Never overridable. A caller cannot ask these backends to reach the network.
LOCAL_FILES_ONLY = True

# Enough for a short constrained JSON decision object; greedy, so it terminates.
DEFAULT_MAX_NEW_TOKENS = 256

# Default open-type prompt labels. GLiNER's label space is prompted, so the
# mapping from these strings into the five organizer types is this project's
# judgement and lives with the L3 adapter
# (``mention_factory.gliner.OPEN_TYPE_LABEL_TO_TYPE``), not here.
DEFAULT_OPEN_TYPE_LABELS: tuple[str, ...] = (
    "medication", "disease", "symptom", "test", "test result")

DEFAULT_OPEN_TYPE_THRESHOLD = 0.35


class BackendError(RuntimeError):
    """Raised when a backend cannot be loaded or run.

    Deliberately loud. Degrading a missing model to "no predictions" would report
    a broken environment as a measured result, which is the worst available
    outcome for an ablation.
    """


class LocalCheckpointMissing(BackendError):
    """Raised when the configured local checkpoint directory is absent.

    This is the fail-closed path Appendix A requires: no fallback to a hub id, no
    download, no partially initialized model.
    """


class SpanBackend(Protocol):
    """Anything that turns text into raw labelled spans with offsets."""

    def predict(self, text: str) -> Sequence[Mapping[str, Any]]: ...


class TextBackend(Protocol):
    """Anything that turns a prompt into a text completion."""

    def generate(self, prompt: str) -> str: ...


def require_local_directory(path: str | Path, *, role: str) -> Path:
    """Resolve a local model directory, or refuse by name.

    The error names the role and the exact missing path, because "model not
    found" without either is unactionable in a fresh runtime.
    """
    if not str(path):
        raise LocalCheckpointMissing(
            f"{role} has no configured local path; this project never resolves a "
            "model by hub id at inference (spec Appendix A)")
    resolved = Path(path)
    if not resolved.is_dir():
        raise LocalCheckpointMissing(
            f"{role} local checkpoint directory does not exist: {resolved}. "
            "Materialize it before inference; nothing is downloaded here.")
    return resolved


@dataclass
class OpenTypeSpanBackend:
    """A local open-type span tagger (spec §6, expert E6).

    Returns the tagger's own offsets. They are **not** trusted by this class:
    ``mention_factory.offsets.verify_span`` re-checks every one against the
    original text and the L3 adapter rejects any that disagree.
    """

    local_path: str
    model_id: str = ""
    revision: str = ""
    threshold: float = DEFAULT_OPEN_TYPE_THRESHOLD
    labels: tuple[str, ...] = DEFAULT_OPEN_TYPE_LABELS
    role: str = "open-type span backend"
    _model: Any = field(default=None, repr=False)

    def load(self) -> OpenTypeSpanBackend:
        path = require_local_directory(self.local_path, role=self.role)
        try:
            from gliner import GLiNER
        except ImportError as error:  # pragma: no cover - optional heavy dependency
            raise BackendError(
                "the 'gliner' package is not installed; the open-type span "
                "backend cannot run without it") from error
        try:
            self._model = GLiNER.from_pretrained(
                str(path), local_files_only=LOCAL_FILES_ONLY)
        except Exception as error:  # noqa: BLE001 - re-raised as a prerequisite
            raise BackendError(
                f"could not load the open-type span backend from {path}: {error}"
            ) from error
        self._model.eval()
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def parameter_count(self) -> int:
        """A real count, for the deployment ledger. Never a published estimate."""
        if self._model is None:
            raise BackendError(
                f"{self.role} is not loaded; cannot count parameters")
        return int(sum(p.numel() for p in self._model.parameters()))

    def predict(self, text: str) -> Sequence[Mapping[str, Any]]:
        if self._model is None:
            raise BackendError(f"{self.role} is not loaded")
        import torch

        with torch.no_grad():
            raw = self._model.predict_entities(
                text, list(self.labels), threshold=self.threshold)
        return [
            {"start": int(item["start"]), "end": int(item["end"]),
             "label": str(item["label"]), "score": float(item.get("score", 0.0))}
            for item in raw
        ]


@dataclass
class CausalLMBackend:
    """One local causal LM, greedy-decoded, serving every L7 cascade duty.

    Spec §12 puts a critic and an adjudicator in the cascade and §17 caps the
    deployment at 9B. Sharing a single instance across mention proposal, assertion
    adjudication and candidate selection is what keeps the *concurrently resident*
    total small — the ledger counts a shared backbone once
    (``governance.parameter_budget``), so a second instance is a real budget cost
    and must be justified rather than assumed.
    """

    local_path: str
    model_id: str = ""
    revision: str = ""
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    role: str = "causal LM backend"
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    def load(self) -> CausalLMBackend:
        path = require_local_directory(self.local_path, role=self.role)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - optional heavy dependency
            raise BackendError(
                "transformers is not installed; the causal LM backend cannot run "
                "without it") from error
        import torch

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(path), local_files_only=LOCAL_FILES_ONLY)
            self._model = AutoModelForCausalLM.from_pretrained(
                str(path),
                local_files_only=LOCAL_FILES_ONLY,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        except Exception as error:  # noqa: BLE001 - re-raised as a prerequisite
            raise BackendError(
                f"could not load the causal LM backend from {path}: {error}"
            ) from error
        if torch.cuda.is_available():
            self._model = self._model.to("cuda")
        self._model.eval()
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def parameter_count(self) -> int:
        """A real count, for the deployment ledger. Never a published estimate."""
        if self._model is None:
            raise BackendError(f"{self.role} is not loaded; cannot count parameters")
        return int(sum(p.numel() for p in self._model.parameters()))

    def generate(self, prompt: str) -> str:
        """Greedy completion. Deterministic by construction: no sampling."""
        if self._model is None or self._tokenizer is None:
            raise BackendError(f"{self.role} is not loaded")
        import torch

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            rendered = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:  # pragma: no cover - tokenizers without a chat template
            rendered = prompt
        encoded = self._tokenizer(rendered, return_tensors="pt")
        encoded = {k: v.to(self._model.device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = self._model.generate(
                **encoded, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=self._tokenizer.eos_token_id)
        completion = generated[0][encoded["input_ids"].shape[-1]:]
        return str(self._tokenizer.decode(completion, skip_special_tokens=True))


def assert_no_external_api(configuration: Mapping[str, Any]) -> None:
    """Every model runs self-hosted; no external API, ever (Appendix A)."""
    configured = [
        key for key in (
            "openai_api_key", "anthropic_api_key", "external_api_base",
            "remote_inference_endpoint")
        if configuration.get(key)
    ]
    if configured:
        raise BackendError(
            f"{sorted(configured)} configured; this project forbids any external "
            "model API and runs every model self-hosted (spec Appendix A)")


__all__ = [
    "BACKEND_CONTRACT_VERSION",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_OPEN_TYPE_LABELS",
    "DEFAULT_OPEN_TYPE_THRESHOLD",
    "LOCAL_FILES_ONLY",
    "BackendError",
    "CausalLMBackend",
    "LocalCheckpointMissing",
    "OpenTypeSpanBackend",
    "SpanBackend",
    "TextBackend",
    "assert_no_external_api",
    "require_local_directory",
]
