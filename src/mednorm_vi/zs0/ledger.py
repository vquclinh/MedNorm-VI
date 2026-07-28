"""ZS0 exact parameter ledger (Audit 0048).

Spec §17 caps a deployment at 9B parameters. This ledger answers the only
question that matters at inference time: **how many parameters are resident at
once?**

Two rules, both learned the hard way in Audit 0042:

* a LoRA deployment counts the base model **plus** its adapters, never the
  adapter alone;
* a component whose count is not verified fails the gate **closed**. An
  unverified estimate is not a number you may submit against a hard cap.

The physical total (everything on disk) and the active total (everything
concurrently resident) are reported separately, because only the second is what
the cap governs — and conflating them either wastes budget or breaks the rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEDGER_CONTRACT_VERSION = "zs0-parameter-ledger-v1"

MAX_ACTIVE_PARAMETERS = 9_000_000_000

VERIFIED_METHODS: frozenset[str] = frozenset(
    {"counted_from_config", "counted_from_checkpoint", "counted_from_safetensors_index"})


class LedgerError(ValueError):
    """Raised when the ledger cannot certify a deployment."""


class BudgetExceeded(LedgerError):
    """Raised when the active total exceeds the 9B cap."""


class UnverifiedComponent(LedgerError):
    """Raised when an active component has no verified parameter count."""


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One model in the ZS0 deployment."""

    component_id: str
    model_id: str
    revision: str
    checkpoint_sha256: str
    parameter_count: int | None
    adapter_parameters: int = 0
    loaded_during_inference: bool = False
    concurrent_with_other_models: bool = False
    shared_instance_key: str = ""
    count_method: str = "unknown"
    count_verified: bool = False

    def __post_init__(self) -> None:
        if self.count_verified and self.count_method not in VERIFIED_METHODS:
            raise LedgerError(
                f"{self.component_id}: count_verified is true but the method is "
                f"{self.count_method!r}; verification needs a real programmatic count")
        if self.count_verified and self.parameter_count is None:
            raise LedgerError(
                f"{self.component_id}: verified with no parameter count")
        if self.parameter_count is not None and self.parameter_count < 0:
            raise LedgerError(f"{self.component_id}: negative parameter count")
        if self.adapter_parameters < 0:
            raise LedgerError(f"{self.component_id}: negative adapter parameters")

    @property
    def physical_total(self) -> int:
        """Base plus adapters. A LoRA deployment is never just its adapter."""
        return int(self.parameter_count or 0) + int(self.adapter_parameters)

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "parameter_count": self.parameter_count,
            "adapter_parameters": self.adapter_parameters,
            "physical_total": self.physical_total,
            "loaded_during_inference": self.loaded_during_inference,
            "concurrent_with_other_models": self.concurrent_with_other_models,
            "shared_instance_key": self.shared_instance_key,
            "count_method": self.count_method,
            "count_verified": self.count_verified,
        }


@dataclass(frozen=True, slots=True)
class LedgerReport:
    """The certified answer, or the reason there isn't one."""

    entries: tuple[LedgerEntry, ...]
    physical_total: int
    active_total: int
    concurrent_total: int
    within_budget: bool
    remaining_margin: int
    unverified_active: tuple[str, ...]
    shared_bases_counted_once: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ledger_contract_version": LEDGER_CONTRACT_VERSION,
            "entries": [entry.as_dict() for entry in self.entries],
            "physical_total_parameters": self.physical_total,
            "active_deployment_total_parameters": self.active_total,
            "concurrently_resident_parameters": self.concurrent_total,
            "max_active_parameters": MAX_ACTIVE_PARAMETERS,
            "within_budget": self.within_budget,
            "remaining_margin": self.remaining_margin,
            "unverified_active_components": list(self.unverified_active),
            "shared_bases_counted_once": list(self.shared_bases_counted_once),
            "adapter_only_counting_used": False,
        }


def build_ledger(
    entries: Sequence[LedgerEntry], *, enforce: bool = True,
) -> LedgerReport:
    """Total the deployment and gate it. Fails closed on anything unverified."""
    physical = sum(entry.physical_total for entry in entries)

    active_entries = [e for e in entries if e.loaded_during_inference]
    unverified = tuple(sorted(
        e.component_id for e in active_entries if not e.count_verified))

    counted_bases: set[str] = set()
    shared_once: list[str] = []
    active_total = 0
    for entry in sorted(active_entries, key=lambda e: e.component_id):
        key = entry.shared_instance_key
        if key and key in counted_bases:
            # The backbone is already resident; only the adapter adds.
            active_total += int(entry.adapter_parameters)
            shared_once.append(key)
            continue
        if key:
            counted_bases.add(key)
        active_total += entry.physical_total

    concurrent_total = sum(
        e.physical_total for e in active_entries if e.concurrent_with_other_models)

    within = active_total <= MAX_ACTIVE_PARAMETERS and not unverified
    report = LedgerReport(
        entries=tuple(entries), physical_total=physical, active_total=active_total,
        concurrent_total=concurrent_total, within_budget=within,
        remaining_margin=MAX_ACTIVE_PARAMETERS - active_total,
        unverified_active=unverified,
        shared_bases_counted_once=tuple(sorted(set(shared_once))))

    if enforce:
        if unverified:
            raise UnverifiedComponent(
                f"active components {list(unverified)} have no verified parameter "
                "count; the 9B gate fails closed rather than trusting an estimate")
        if active_total > MAX_ACTIVE_PARAMETERS:
            raise BudgetExceeded(
                f"active deployment total {active_total:,} exceeds the "
                f"{MAX_ACTIVE_PARAMETERS:,} cap by "
                f"{active_total - MAX_ACTIVE_PARAMETERS:,}")
    return report


def render_ledger(report: LedgerReport) -> str:
    """A compact table for the audit."""
    header = ("| component | model | params | adapter | physical | active? | "
              "concurrent? | shared | verified |")
    rule = "| " + " | ".join(["---"] * 9) + " |"
    rows = [
        f"| {e.component_id} | {e.model_id} | "
        f"{e.parameter_count if e.parameter_count is not None else 'UNVERIFIED'} | "
        f"{e.adapter_parameters} | {e.physical_total} | "
        f"{e.loaded_during_inference} | {e.concurrent_with_other_models} | "
        f"{e.shared_instance_key or '-'} | {e.count_verified} |"
        for e in report.entries
    ]
    total = (f"\nactive deployment total: {report.active_total:,} / "
             f"{MAX_ACTIVE_PARAMETERS:,}  (margin {report.remaining_margin:,}, "
             f"within budget {report.within_budget})")
    return "\n".join([header, rule, *rows]) + total


def load_ledger(path: str | Path) -> tuple[LedgerEntry, ...]:
    """Read a tracked ledger YAML into entries."""
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries: list[LedgerEntry] = []
    for raw in document.get("components", []):
        count = raw.get("parameter_count")
        entries.append(LedgerEntry(
            component_id=str(raw["component_id"]),
            model_id=str(raw.get("model_id", "")),
            revision=str(raw.get("revision", "")),
            checkpoint_sha256=str(raw.get("checkpoint_sha256", "")),
            parameter_count=None if count is None else int(count),
            adapter_parameters=int(raw.get("adapter_parameters", 0)),
            loaded_during_inference=bool(raw.get("loaded_during_inference", False)),
            concurrent_with_other_models=bool(
                raw.get("concurrent_with_other_models", False)),
            shared_instance_key=str(raw.get("shared_instance_key", "")),
            count_method=str(raw.get("count_method", "unknown")),
            count_verified=bool(raw.get("count_verified", False))))
    return tuple(entries)


__all__ = [
    "LEDGER_CONTRACT_VERSION",
    "MAX_ACTIVE_PARAMETERS",
    "VERIFIED_METHODS",
    "BudgetExceeded",
    "LedgerEntry",
    "LedgerError",
    "LedgerReport",
    "UnverifiedComponent",
    "build_ledger",
    "load_ledger",
    "render_ledger",
]
