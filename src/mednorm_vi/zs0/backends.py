"""Real pretrained inference backends for ZS0 (Audit 0049).

Audit 0048 shipped ZS0's *contracts* — offset verification, substring
resolution, candidate constraints — but never the code that actually calls a
model. Stage 3 was a placeholder, so nothing ever produced a proposal. These are
the real inference paths.

Both backends import their heavy dependency **lazily**, inside `load()`. The
module therefore imports on a laptop with neither `gliner` nor a GPU present,
which is what lets the contract tests run locally while the models run on Colab.

Nothing here trains: no optimizer, no backward pass, and every forward runs under
`torch.no_grad()` with the model in `eval()`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

BACKEND_CONTRACT_VERSION = "zs0-backend-v1"

# The prompted GLiNER label set. These are the strings the model is asked for;
# `proposals.GLINER_LABEL_TO_TYPE` maps them into the five BTC types.
GLINER_PROMPT_LABELS: tuple[str, ...] = (
    "medication", "disease", "symptom", "test", "test result")

DEFAULT_GLINER_THRESHOLD = 0.35
DEFAULT_GLINER_MODEL = "urchade/gliner_medium-v2.1"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-1.7B"
# Enough for a short JSON object of mentions; greedy, so it terminates.
DEFAULT_QWEN_MAX_NEW_TOKENS = 256


class BackendError(RuntimeError):
    """Raised when a backend cannot be loaded or run.

    Deliberately loud: a missing model is a prerequisite failure, and silently
    degrading an arm to "no predictions" would report a broken environment as a
    measured result.
    """


class SpanBackend(Protocol):
    """Anything that turns text into raw labelled spans."""

    def predict(self, text: str) -> Sequence[Mapping[str, Any]]: ...


class TextBackend(Protocol):
    """Anything that turns a prompt into a text completion."""

    def generate(self, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# GLiNER
# ---------------------------------------------------------------------------


@dataclass
class GLiNERBackend:
    """Pretrained GLiNER, self-hosted. No fine-tuning, no external API."""

    model_id: str = DEFAULT_GLINER_MODEL
    revision: str = ""
    threshold: float = DEFAULT_GLINER_THRESHOLD
    labels: tuple[str, ...] = GLINER_PROMPT_LABELS
    cache_dir: str = ""
    _model: Any = field(default=None, repr=False)

    def load(self) -> GLiNERBackend:
        try:
            from gliner import GLiNER
        except ImportError as error:  # pragma: no cover - Colab-only dependency
            raise BackendError(
                "the 'gliner' package is not installed; ZS0-B and ZS0-C cannot "
                "run without it. Stage 0 installs it."
            ) from error
        kwargs: dict[str, Any] = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        try:
            self._model = GLiNER.from_pretrained(self.model_id, **kwargs)
        except Exception as error:  # noqa: BLE001 - re-raised as a prerequisite
            raise BackendError(
                f"could not load GLiNER {self.model_id!r}: {error}") from error
        self._model.eval()
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def parameter_count(self) -> int:
        if self._model is None:
            raise BackendError("GLiNER is not loaded; cannot count parameters")
        return int(sum(p.numel() for p in self._model.parameters()))

    def predict(self, text: str) -> Sequence[Mapping[str, Any]]:
        """Raw spans with the model's own offsets.

        The offsets are **not** trusted here; `proposals.convert_gliner_spans`
        verifies each against the original text and rejects any that disagree.
        """
        if self._model is None:
            raise BackendError("GLiNER is not loaded")
        import torch

        with torch.no_grad():
            raw = self._model.predict_entities(
                text, list(self.labels), threshold=self.threshold)
        return [
            {"start": int(item["start"]), "end": int(item["end"]),
             "label": str(item["label"]), "score": float(item.get("score", 0.0))}
            for item in raw
        ]


# ---------------------------------------------------------------------------
# Qwen
# ---------------------------------------------------------------------------

QWEN_MENTION_INSTRUCTION = (
    "You extract clinical mentions from Vietnamese text.\n"
    "Return ONLY a JSON object of the form "
    '{"mentions": [{"text": "...", "type": "..."}]}.\n'
    "Rules:\n"
    "- every \"text\" MUST be copied verbatim from the segment; never paraphrase, "
    "translate, correct or invent;\n"
    "- \"type\" MUST be one of MEDICATION, DIAGNOSIS, SYMPTOM, TEST_NAME, "
    "TEST_RESULT;\n"
    "- if a mention appears more than once, add an \"anchor\" field containing a "
    "longer verbatim substring that contains the intended occurrence exactly once;\n"
    "- return {\"mentions\": []} when unsure. An empty list is a correct answer.\n"
    "Output no prose, no markdown fence, no explanation."
)


@dataclass
class QwenBackend:
    """One pretrained Qwen instance, greedy-decoded, serving every ZS0 duty.

    Sharing one instance across mention proposal, assertion adjudication and
    candidate selection is what keeps the active parameter total small enough
    that a larger model never has to be added to the ledger.
    """

    model_id: str = DEFAULT_QWEN_MODEL
    revision: str = ""
    max_new_tokens: int = DEFAULT_QWEN_MAX_NEW_TOKENS
    cache_dir: str = ""
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    def load(self) -> QwenBackend:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - Colab-only dependency
            raise BackendError(
                "transformers is not installed; ZS0-C cannot run without it"
            ) from error
        import torch

        kwargs: dict[str, Any] = {}
        if self.revision:
            kwargs["revision"] = self.revision
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                **kwargs)
        except Exception as error:  # noqa: BLE001 - re-raised as a prerequisite
            raise BackendError(
                f"could not load Qwen {self.model_id!r}: {error}") from error
        if torch.cuda.is_available():
            self._model = self._model.to("cuda")
        self._model.eval()
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def parameter_count(self) -> int:
        if self._model is None:
            raise BackendError("Qwen is not loaded; cannot count parameters")
        return int(sum(p.numel() for p in self._model.parameters()))

    def generate(self, prompt: str) -> str:
        """Greedy completion. Deterministic by construction: no sampling."""
        if self._model is None or self._tokenizer is None:
            raise BackendError("Qwen is not loaded")
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


def build_mention_prompt(segment: str) -> str:
    """The exact prompt Qwen sees for hard mention proposal."""
    return f"{QWEN_MENTION_INSTRUCTION}\n\nSegment:\n{segment}\n\nJSON:"


def extract_json_object(completion: str) -> str:
    """Recover the first balanced JSON object from a completion.

    Instruction-tuned models add a fence or a sentence however firmly they are
    told not to. Extracting the object is not repairing the content: anything
    that is not a well-formed object still fails, and the caller's schema and
    substring checks still apply to whatever comes out.
    """
    start = completion.find("{")
    if start < 0:
        return completion
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(completion)):
        character = completion[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return completion[start:index + 1]
    return completion


def parse_completion(completion: str) -> str:
    """Normalize a completion to a JSON string the ZS0 parsers can consume."""
    candidate = extract_json_object(completion.strip())
    try:
        json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        # Hand the raw text through; the caller rejects it with a reason code.
        return candidate
    return candidate


__all__ = [
    "BACKEND_CONTRACT_VERSION",
    "DEFAULT_GLINER_MODEL",
    "DEFAULT_GLINER_THRESHOLD",
    "DEFAULT_QWEN_MAX_NEW_TOKENS",
    "DEFAULT_QWEN_MODEL",
    "GLINER_PROMPT_LABELS",
    "QWEN_MENTION_INSTRUCTION",
    "BackendError",
    "GLiNERBackend",
    "QwenBackend",
    "SpanBackend",
    "TextBackend",
    "build_mention_prompt",
    "extract_json_object",
    "parse_completion",
]
