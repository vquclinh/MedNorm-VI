"""PositionPolicyRegistry — load registered policies and encode raw spans.

Reads ``configs/organizer/position_policies_v1.yaml``. The default policy is the
internal / diagnostic ``raw-codepoint-half-open``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..schemas.spans import OffsetAlignment
from .encoders import PositionEncodingError, decode_position, encode_span
from .models import PositionEncodingResult, PositionPolicy


class PositionPolicyRegistry:
    """Registered position policies with deterministic encode/decode access."""

    def __init__(self, policies: list[PositionPolicy], default_policy_id: str,
                 config_hash: str) -> None:
        self._policies = {p.policy_id: p for p in policies}
        if default_policy_id not in self._policies:
            raise ValueError(f"default policy {default_policy_id!r} is not registered")
        self.default_policy_id = default_policy_id
        self.config_hash = config_hash

    @property
    def policy_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._policies))

    def policy(self, policy_id: str) -> PositionPolicy:
        if policy_id not in self._policies:
            raise KeyError(f"unknown position policy {policy_id!r}")
        return self._policies[policy_id]

    def encode(
        self, policy_id: str, text: str, start: int, end: int, *,
        separator: str = "\n", alignment: OffsetAlignment | None = None,
    ) -> PositionEncodingResult:
        return encode_span(self.policy(policy_id), text, start, end,
                           separator=separator, alignment=alignment)

    def decode(
        self, policy_id: str, text: str, enc_start: int, enc_end: int, *,
        separator: str = "\n", alignment: OffsetAlignment | None = None,
    ) -> tuple[int, int]:
        return decode_position(self.policy(policy_id), text, enc_start, enc_end,
                               separator=separator, alignment=alignment)

    def round_trips(
        self, policy_id: str, text: str, start: int, end: int, *,
        separator: str = "\n", alignment: OffsetAlignment | None = None,
    ) -> bool:
        """True iff encode-then-decode recovers the exact raw span."""
        try:
            enc = self.encode(policy_id, text, start, end, separator=separator,
                              alignment=alignment)
            back = self.decode(policy_id, text, enc.start, enc.end, separator=separator,
                               alignment=alignment)
        except PositionEncodingError:
            return False
        return back == (start, end)


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_position_registry(config_path: str | Path) -> PositionPolicyRegistry:
    """Load the position-policy registry from a YAML config file."""
    import yaml

    path = Path(config_path)
    doc: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    policies = [
        PositionPolicy(
            policy_id=str(p["policy_id"]), title=str(p.get("title", p["policy_id"])),
            coordinate_space=str(p["coordinate_space"]), interval=str(p["interval"]),
            line_ending=str(p["line_ending"]), reversible=str(p.get("reversible", "true")),
            description=str(p.get("description", "")))
        for p in doc.get("policies", []) or []
    ]
    default = str(doc.get("default_policy", "raw-codepoint-half-open"))
    return PositionPolicyRegistry(policies, default, _hash(doc))


__all__ = ["PositionPolicyRegistry", "load_position_registry"]
