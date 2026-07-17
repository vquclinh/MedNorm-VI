"""Legacy/current RXCUI compatibility over a local snapshot (Phase 1C-A).

Resolves a possibly-retired RXCUI to a current one by following remap relations.
This is a provisional, documented heuristic — whether the organizer expects
remapping at all is UNRESOLVED (UNRES-RXNORM-REMAP); nothing here decides it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import RxnormSnapshot


@dataclass(frozen=True, slots=True)
class RemapResolution:
    """Result of resolving a (possibly legacy) RXCUI to a current concept."""

    query_rxcui: str
    resolved_rxcui: str
    chain: tuple[str, ...]  # query -> ... -> resolved
    was_remapped: bool
    resolved_is_suppressed: bool
    resolved_exists: bool


def resolve_current(snapshot: RxnormSnapshot, rxcui: str, *, max_hops: int = 8) -> RemapResolution:
    """Follow remap targets deterministically to a stable current RXCUI."""
    chain = [rxcui]
    current = rxcui
    seen = {rxcui}
    for _ in range(max_hops):
        targets = snapshot.remap_targets(current)
        # pick the smallest target id for determinism; stop on cycle/none
        nxt = next((t for t in targets if t not in seen), None)
        if nxt is None:
            break
        chain.append(nxt)
        seen.add(nxt)
        current = nxt
    exists = bool(snapshot.atoms_for(current))
    return RemapResolution(
        query_rxcui=rxcui, resolved_rxcui=current, chain=tuple(chain),
        was_remapped=current != rxcui, resolved_is_suppressed=snapshot.is_suppressed(current),
        resolved_exists=exists)


def is_active(snapshot: RxnormSnapshot, rxcui: str) -> bool:
    """True when the concept exists and is not fully suppressed."""
    return bool(snapshot.atoms_for(rxcui)) and not snapshot.is_suppressed(rxcui)


__all__ = ["RemapResolution", "resolve_current", "is_active"]
