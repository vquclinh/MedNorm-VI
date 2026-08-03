"""The deployed Qwen3-8B host: greedy, bounded, loaded on demand and released.

One model object serves every stage that needs a language model - contextual proposal,
finite verification, query reformulation and set-wise reranking - so the deployment contains
exactly one 8B model however many times it is asked a question.

`ask_detailed` exists because structured extraction has to know whether generation stopped
because it ran out of budget: a reply cut off mid-JSON is indistinguishable from a bad model
unless the caller is told.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def log(message: str) -> None:
    print(f"[mednorm] {message}", flush=True)


def parameters_from_safetensors(root: Path) -> int:
    """Sum tensor elements from safetensors headers alone. 0 when there is nothing to read.

    Each shard begins with an 8-byte little-endian header length followed by a JSON header
    mapping tensor name to dtype/shape/offsets. Reading shapes needs no weights in memory,
    and tied weights are stored once, exactly as `model.parameters()` counts them once.
    """
    total = 0
    shards = sorted(Path(root).glob("*.safetensors")) if Path(root).is_dir() else []
    for shard in shards:
        with shard.open("rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            if length <= 0:
                continue
            header = json.loads(handle.read(length).decode("utf-8"))
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            count = 1
            for dim in meta.get("shape") or []:
                count *= int(dim)
            total += count if (meta.get("shape") or []) else 0
    return total


class Disambiguator(Protocol):
    """Anything that answers one prompt. Injected so tests need no weights."""

    def ask(self, prompt: str) -> str: ...

    def unload(self) -> None: ...


@dataclass
class QwenDisambiguator:
    """The real local Qwen3-8B. Greedy, bounded, loaded lazily."""

    model_root: Path
    device: str = "cuda"
    max_new_tokens: int = 256
    _model: Any = None
    _tokenizer: Any = None
    system_prompt: str = (
        "You are a precise clinical entity linker. You return STRICT JSON and nothing else. "
        "You select candidate indices only; you never write a medical code."
    )

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = str(self.model_root)
        self._tokenizer = AutoTokenizer.from_pretrained(source)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._model = (
            AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        config = self._model.generation_config
        config.do_sample = False
        for name in ("temperature", "top_p", "top_k"):
            if hasattr(config, name):
                setattr(config, name, None)
        log("Qwen3-8B loaded (greedy, sampling parameters cleared)")

    def parameter_count(self) -> int:
        """Exact count, preferably without materializing 8B parameters.

        The safetensors header carries every tensor's shape, so the budget can be certified
        on a host that could not hold the model - which matters because `download-models`
        only requires enough memory for a retriever. Loading is the fallback, not the path.
        """
        counted = parameters_from_safetensors(self.model_root)
        if counted:
            return counted
        self.load()
        return sum(p.numel() for p in self._model.parameters())

    def ask_detailed(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """`ask`, plus the shape of what was generated. Added, never substituted.

        0080 calls `ask` and is unaffected. 0081 needs to know whether generation stopped
        because it ran out of budget: a structured extraction that silently hits its token
        limit produces an unclosed JSON object, which is indistinguishable from a bad model
        unless the caller is told. That is what hid the first proposer failure.
        """
        reply, stats = self._generate(prompt)
        return reply, stats

    def ask(self, prompt: str) -> str:
        return self._generate(prompt)[0]

    def _generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        import torch

        self.load()
        text = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        encoded = self._tokenizer([text], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        produced = generated[0][encoded["input_ids"].shape[1] :]
        reply = self._tokenizer.decode(produced, skip_special_tokens=True)
        return str(reply), {
            "generated_tokens": int(produced.shape[0]),
            "hit_token_limit": int(produced.shape[0]) >= int(self.max_new_tokens),
            "max_new_tokens": int(self.max_new_tokens),
        }

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass


__all__ = [
    "Disambiguator",
    "QwenDisambiguator",
    "log",
    "parameters_from_safetensors",
]
