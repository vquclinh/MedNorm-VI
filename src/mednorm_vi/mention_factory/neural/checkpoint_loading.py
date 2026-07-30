"""How a checkpoint is allowed to be deserialized (Audit 0056a, finding I).

``torch.load(..., weights_only=False)`` executes arbitrary pickle opcodes. The E3
path has always verified the checkpoint's SHA-256 **before** loading, so this was
never an open door — but the ordering was a convention held by one call site, and
nothing stopped a future checkpoint from being loaded the same way with no digest at
all. A guarantee that depends on remembering to hash first is not a guarantee.

This module makes the rule structural, and splits one policy into two:

**The trusted-legacy path.** E3's ``best.pt`` is a torch-pickled artifact produced
before this project adopted safetensors. It carries non-tensor metadata the loader
must read (``mode``, ``entity_type_order``, ``pinned_model_revision``), so
``weights_only=True`` cannot load it as-is. It is admitted by *identity*: only a file
whose digest equals :data:`TRUSTED_LEGACY_E3_SHA256` may be opened with
``weights_only=False``, and the allowance is named, dated and limited to that one
artifact rather than being a property of the loader.

**The default path, for every future checkpoint.** Safe loading
(``weights_only=True``) or safetensors. A new checkpoint that wants the legacy
treatment does not get it: :func:`assert_load_is_permitted` refuses, and names
safetensors as the fix. This is what stops the legacy allowance from quietly becoming
the house style.

Two invariants, both enforced here rather than documented:

1. **an unverified checkpoint can never reach ``weights_only=False``** — the digest
   is required, not optional, on that path;
2. **the digest is checked before any deserialization** — the caller passes an
   already-computed fingerprint, so there is no ordering left to get wrong.

Nothing in this module loads anything. It decides whether a load is permitted; the
caller performs it. That is what lets 2D-A test the policy without importing torch
and without touching the real 1.6 GB artifact. Runtime verification that the E3
forward pass still produces identical output is deferred to Milestone 2D-C, where
model loading is authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: The one artifact admitted to the legacy pickle path. This is E3's validated
#: `best.pt` (Audits 0031, 0032, 0052). Any other file, including a *newer* E3
#: checkpoint, must use safe loading or safetensors.
TRUSTED_LEGACY_E3_SHA256 = "a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c"

#: Suffixes that are safe by format: the reader never executes code.
SAFETENSORS_SUFFIXES: tuple[str, ...] = (".safetensors",)

LOAD_MODE_TRUSTED_LEGACY_PICKLE = "TRUSTED_LEGACY_PICKLE"
LOAD_MODE_WEIGHTS_ONLY = "WEIGHTS_ONLY"
LOAD_MODE_SAFETENSORS = "SAFETENSORS"


class CheckpointTrustError(RuntimeError):
    """Raised when a checkpoint may not be deserialized the way a caller asked."""


@dataclass(frozen=True, slots=True)
class LoadPolicy:
    """The decision: how this specific checkpoint may be read."""

    mode: str
    weights_only: bool
    reason: str

    @property
    def is_trusted_legacy(self) -> bool:
        return self.mode == LOAD_MODE_TRUSTED_LEGACY_PICKLE


def is_safetensors(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SAFETENSORS_SUFFIXES


def assert_load_is_permitted(
    path: str | Path,
    *,
    verified_sha256: str,
    expected_sha256: str = "",
    trusted_legacy_sha256: str = TRUSTED_LEGACY_E3_SHA256,
) -> LoadPolicy:
    """Decide how ``path`` may be deserialized. Loads nothing; raises when unsafe.

    ``verified_sha256`` is the digest the caller **already computed** from the file
    on disk. Requiring it as an argument is the point: a caller cannot reach the
    legacy path without having hashed the bytes first.

    ``expected_sha256`` is the digest the configuration pinned, when there is one.
    A pinned digest that disagrees with the computed one is refused here as well as
    at fingerprint time — the same fact checked twice, cheaply, because this is the
    check that stands between a substituted file and arbitrary code execution.
    """
    target = Path(path)
    computed = (verified_sha256 or "").strip().lower()
    pinned = (expected_sha256 or "").strip().lower()
    legacy = (trusted_legacy_sha256 or "").strip().lower()

    if pinned and computed and pinned != computed:
        raise CheckpointTrustError(
            f"refusing to load {target}: the pinned digest {pinned} does not match "
            f"the digest of the file on disk ({computed})"
        )

    if is_safetensors(target):
        return LoadPolicy(
            mode=LOAD_MODE_SAFETENSORS,
            weights_only=True,
            reason="safetensors executes no code on read",
        )

    if not computed:
        raise CheckpointTrustError(
            f"refusing to load {target}: no verified SHA-256 was supplied. A pickled "
            "checkpoint is only ever opened after its digest has been computed and "
            "matched, so an unverified file can never reach weights_only=False"
        )

    if legacy and computed == legacy:
        return LoadPolicy(
            mode=LOAD_MODE_TRUSTED_LEGACY_PICKLE,
            weights_only=False,
            reason=(
                "digest matches the one trusted legacy artifact (E3 best.pt, Audits "
                "0031/0032/0052); it carries non-tensor metadata that weights_only "
                "cannot read, and the allowance is limited to this exact file"
            ),
        )

    return LoadPolicy(
        mode=LOAD_MODE_WEIGHTS_ONLY,
        weights_only=True,
        reason=(
            "an unrecognised checkpoint is loaded with weights_only=True; convert it "
            "to safetensors, or keep tensors and metadata in separate files, rather "
            "than extending the legacy pickle allowance"
        ),
    )


def describe_policy(policy: LoadPolicy) -> dict[str, object]:
    """Provenance for the run manifest. Carries no path and no clinical text."""
    return {
        "load_mode": policy.mode,
        "weights_only": policy.weights_only,
        "trusted_legacy": policy.is_trusted_legacy,
    }


__all__ = [
    "LOAD_MODE_SAFETENSORS",
    "LOAD_MODE_TRUSTED_LEGACY_PICKLE",
    "LOAD_MODE_WEIGHTS_ONLY",
    "SAFETENSORS_SUFFIXES",
    "TRUSTED_LEGACY_E3_SHA256",
    "CheckpointTrustError",
    "LoadPolicy",
    "assert_load_is_permitted",
    "describe_policy",
    "is_safetensors",
]
