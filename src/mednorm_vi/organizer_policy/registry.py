"""Load the organizer-policy registry from ``configs/organizer/*.yaml``.

Deterministic, offline, read-only. Produces an :class:`OrganizerPolicyRegistry`
of confirmed facts + unresolved/hypothesised policies, plus a stable
``config_hash`` for the doctor CLI determinism check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import (
    ConfirmedFact,
    OrganizerPolicyRegistry,
    PolicyHypothesis,
    PolicyOption,
)

_CONFIRMED = "confirmed_facts_v1.yaml"
_UNRESOLVED = "unresolved_policies_v1.yaml"
_POSITION = "position_policies_v1.yaml"
_RXNORM = "rxnorm_decoding_hypotheses_v1.yaml"
_ICD = "icd_format_hypotheses_v1.yaml"
_HISTORICAL = "historical_policies_v1.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _fact(d: dict[str, Any]) -> ConfirmedFact:
    return ConfirmedFact(
        fact_id=str(d["fact_id"]), statement=str(d["statement"]).strip(),
        source=str(d.get("source", "")),
        note=(str(d["note"]).strip() if d.get("note") is not None else None),
        since=(str(d["since"]) if d.get("since") is not None else None))


def _hypothesis(d: dict[str, Any], registry: str) -> PolicyHypothesis:
    options = tuple(
        PolicyOption(id=str(o["id"]), description=str(o.get("description", "")))
        for o in (d.get("options", []) or [])
    )
    lb = d.get("leaderboard_experiment_id")
    return PolicyHypothesis(
        policy_id=str(d["policy_id"]),
        title=str(d.get("title", d["policy_id"])),
        status=str(d.get("status", "unresolved")),
        question=str(d.get("question", "")).strip(),
        description=str(d.get("description", "")).strip(),
        options=options,
        supporting_observation=str(d.get("supporting_observation", "")).strip(),
        contradicting_evidence=str(d.get("contradicting_evidence", "")).strip(),
        confidence=str(d.get("confidence", "low")),
        test_method=str(d.get("test_method", "")).strip(),
        leaderboard_experiment_id=(str(lb) if lb is not None else None),
        internal_default=(str(d["internal_default"]) if d.get("internal_default") is not None
                          else None),
        linked_unresolved=(str(d["linked_unresolved"]) if d.get("linked_unresolved") is not None
                           else None),
        registry=registry)


def load_organizer_registry(configs_dir: str | Path) -> OrganizerPolicyRegistry:
    """Load every organizer-policy config under ``configs_dir`` deterministically."""
    root = Path(configs_dir)
    raw: dict[str, Any] = {}

    confirmed_doc = _load_yaml(root / _CONFIRMED)
    raw[_CONFIRMED] = confirmed_doc
    facts = tuple(_fact(f) for f in confirmed_doc.get("facts", []) or [])

    unresolved_doc = _load_yaml(root / _UNRESOLVED)
    raw[_UNRESOLVED] = unresolved_doc
    unresolved = tuple(_hypothesis(p, "unresolved")
                       for p in unresolved_doc.get("policies", []) or [])

    position_doc = _load_yaml(root / _POSITION)
    raw[_POSITION] = position_doc
    position_ids = tuple(str(p["policy_id"]) for p in position_doc.get("policies", []) or [])
    default_position = str(position_doc.get("default_policy", "raw-codepoint-half-open"))

    rxnorm_doc = _load_yaml(root / _RXNORM)
    raw[_RXNORM] = rxnorm_doc
    rxnorm = tuple(_hypothesis(h, "rxnorm_decoding")
                   for h in rxnorm_doc.get("hypotheses", []) or [])

    icd_doc = _load_yaml(root / _ICD)
    raw[_ICD] = icd_doc
    icd = tuple(_hypothesis(h, "icd_format") for h in icd_doc.get("hypotheses", []) or [])

    hist_doc = _load_yaml(root / _HISTORICAL)
    raw[_HISTORICAL] = hist_doc
    historical = tuple(_hypothesis(h, "historical")
                       for h in hist_doc.get("hypotheses", []) or [])

    return OrganizerPolicyRegistry(
        confirmed_facts=facts,
        unresolved_policies=unresolved,
        position_policy_ids=position_ids,
        default_position_policy=default_position,
        rxnorm_decoding_hypotheses=rxnorm,
        icd_format_hypotheses=icd,
        historical_hypotheses=historical,
        config_hash=_hash(raw))


__all__ = ["load_organizer_registry"]
