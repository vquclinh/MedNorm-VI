"""0081 orchestration: propose, verify, resolve, assert (0081).

One document goes through a fixed pipeline, and every stage is deterministic except the two
Qwen calls, both of which are constrained to choosing among things that already exist:

    E3 + Qwen-context + governed-alias proposals
      -> exact-source alignment          (offsets computed here, never supplied)
      -> finite overlap/boundary lattice (a closed set of alternatives)
      -> Qwen finite verifier            (accept / reject / retype, by index)
      -> deterministic span+type resolver
      -> contextual assertion pass
      -> GraphCENT 0080 candidate linking (unchanged)

Qwen3-8B here is the **same deployed model** GraphCENT already loads. No second base model
enters the deployment, so the parameter accounting is unchanged.

Three variants are emitted from one pass so the leaderboard can attribute any change:
`e3_control` reproduces the previous entity set exactly, `contextual_high` adds only
contextual entities with independent agreement, and `contextual_broad` adds everything the
verifier accepted.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..reasoner.validator import ASSERTION_TYPES, ORGANIZER_TYPES
from .aliases import AliasLexicon, find_alias_hits
from .assertions import run_assertions
from .document import DocumentView
from .lattice import build_lattice
from .proposals import (
    SOURCE_ALIAS,
    SOURCE_E3,
    SOURCE_QWEN,
    Proposal,
    ProposalPool,
    propose,
    propose_from_offsets,
)
from .proposer import build_proposal_prompt, parse_proposals
from .resolver import organizer_entity, resolve
from .verifier import apply_verdicts, build_verification_prompt, parse_verdicts

VARIANT_CONTROL = "e3_control"
VARIANT_HIGH = "contextual_high"
VARIANT_BROAD = "contextual_broad"
VARIANTS: tuple[str, ...] = (VARIANT_CONTROL, VARIANT_HIGH, VARIANT_BROAD)


def log(message: str) -> None:
    print(f"[0081] {message}", flush=True)


class Responder(Protocol):
    """The deployed Qwen3-8B, or a fake in tests. Same object for both passes."""

    def ask(self, prompt: str) -> str: ...


@dataclass(frozen=True, slots=True)
class ContextualConfig:
    """Everything the pass needs. No model paths here - the model is injected."""

    use_qwen_proposer: bool = True
    use_alias_proposer: bool = True
    use_verifier: bool = True
    context_radius: int = 200
    max_options_per_group: int = 12


@dataclass
class DocumentOutcome:
    """One document's final entities plus everything needed to explain them."""

    document: str
    entities: list[Proposal] = field(default_factory=list)
    assertions: dict[tuple[int, int], tuple[str, ...]] = field(default_factory=dict)
    pool: ProposalPool = field(default_factory=ProposalPool)
    counters: dict[str, int] = field(default_factory=dict)

    def count(self, reason: str, amount: int = 1) -> None:
        if amount:
            self.counters[reason] = self.counters.get(reason, 0) + amount

    def merge(self, counters: dict[str, int]) -> None:
        for key, value in counters.items():
            self.count(key, value)


def collect_proposals(
    document: DocumentView,
    seed_entities: Sequence[dict[str, Any]],
    *,
    qwen: Responder | None,
    lexicon: AliasLexicon | None,
    config: ContextualConfig,
    outcome: DocumentOutcome,
) -> ProposalPool:
    """Every source, aligned to the source text. Sources stay attributable."""
    pool = outcome.pool

    for entity in seed_entities:
        position = entity.get("position") or []
        if len(position) != 2:
            pool.reject("seed_missing_position")
            continue
        propose_from_offsets(
            pool, document, source=SOURCE_E3,
            start=int(position[0]), end=int(position[1]),
            entity_type=str(entity.get("type", "")),
        )

    if config.use_qwen_proposer and qwen is not None:
        raw, counters = parse_proposals(qwen.ask(build_proposal_prompt(document)))
        outcome.merge(counters)
        outcome.count("qwen_raw_proposals", len(raw))
        for row in raw:
            propose(
                pool, document, source=SOURCE_QWEN, line_id=row.line_id,
                text=row.text, entity_type=row.type, occurrence=row.occurrence,
            )

    if config.use_alias_proposer and lexicon is not None:
        hits = find_alias_hits(document.source, lexicon)
        outcome.count("alias_hits", len(hits))
        for hit in hits:
            propose_from_offsets(
                pool, document, source=SOURCE_ALIAS,
                start=hit.start, end=hit.end, entity_type=hit.entity_type,
            )

    return pool


def verify_pool(
    document: DocumentView,
    pool: ProposalPool,
    *,
    qwen: Responder | None,
    config: ContextualConfig,
    outcome: DocumentOutcome,
) -> list[Proposal]:
    """Run the finite verifier over each lattice group. Returns accepted proposals."""
    lattice = build_lattice(document, pool.sorted())
    outcome.count("lattice_groups", len(lattice.groups))
    outcome.count("lattice_options", lattice.option_count)
    outcome.count("boundary_unions_offered", lattice.bridges_offered)

    accepted: list[Proposal] = []
    for group in lattice.groups:
        if not config.use_verifier or qwen is None:
            accepted.extend(group.options if group.single else _e3_only(group.options))
            continue
        if len(group.options) > config.max_options_per_group:
            # A group this large is a boundary explosion, not a decision. Fall back to the
            # trusted source rather than asking the model to rank a dozen near-identical
            # spans.
            outcome.count("group_too_large")
            accepted.extend(_e3_only(group.options))
            continue
        reply = qwen.ask(
            build_verification_prompt(document, group, context_radius=config.context_radius)
        )
        verdicts, counters = parse_verdicts(reply, len(group.options))
        outcome.merge(counters)
        result = apply_verdicts(
            group, verdicts, parse_failed=bool(counters.get("verifier_parse_failed"))
        )
        outcome.merge(result.counters)
        outcome.count("verifier_accepted", len(result.accepted))
        outcome.count("verifier_rejected", len(result.rejected))
        outcome.count("verifier_retyped", result.retyped)
        accepted.extend(result.accepted)
    return accepted


def _e3_only(options: Sequence[Proposal]) -> list[Proposal]:
    return [o for o in options if SOURCE_E3 in o.sources]


def process_document(
    name: str,
    source: str,
    seed_entities: Sequence[dict[str, Any]],
    *,
    qwen: Responder | None = None,
    lexicon: AliasLexicon | None = None,
    config: ContextualConfig | None = None,
) -> DocumentOutcome:
    """The whole 0081 pass for one document."""
    settings = config or ContextualConfig()
    document = DocumentView.build(source)
    outcome = DocumentOutcome(document=name)

    pool = collect_proposals(
        document, seed_entities, qwen=qwen, lexicon=lexicon,
        config=settings, outcome=outcome,
    )
    for source_name in (SOURCE_E3, SOURCE_QWEN, SOURCE_ALIAS):
        outcome.count(f"proposals_{source_name}", len(pool.by_source(source_name)))
    outcome.count("proposals_agreed", sum(1 for p in pool.proposals if p.agreed))
    outcome.merge(pool.rejected)

    accepted = verify_pool(
        document, pool, qwen=qwen, config=settings, outcome=outcome
    )
    resolution = resolve(document, accepted)
    outcome.merge(resolution.counters)
    outcome.entities = resolution.entities

    assertions = run_assertions(document, outcome.entities)
    outcome.assertions = assertions.assertions
    outcome.merge(assertions.counters)

    for entity_type in ORGANIZER_TYPES:
        outcome.count(
            f"final_{entity_type}", sum(1 for e in outcome.entities if e.type == entity_type)
        )
    return outcome


# ------------------------------------------------------------------ variants


def variant_entities(outcome: DocumentOutcome, variant: str) -> list[Proposal]:
    """The entity subset one variant emits. All three come from the single pass."""
    if variant == VARIANT_CONTROL:
        return [e for e in outcome.entities if SOURCE_E3 in e.sources]
    if variant == VARIANT_HIGH:
        return [
            e for e in outcome.entities
            if SOURCE_E3 in e.sources or e.agreed
        ]
    return list(outcome.entities)


def serialize(outcome: DocumentOutcome, variant: str) -> list[dict[str, Any]]:
    """Organizer JSON for one variant, in document order, with no unsupported field."""
    rows: list[dict[str, Any]] = []
    for entity in sorted(variant_entities(outcome, variant), key=lambda e: e.start):
        assertions = (
            outcome.assertions.get((entity.start, entity.end), ())
            if entity.type in ASSERTION_TYPES
            else None
        )
        rows.append(organizer_entity(entity, assertions))
    return rows


def write_variants(run_dir: Path, outcomes: list[DocumentOutcome]) -> dict[str, int]:
    """One directory per variant, plus the per-document proposal record."""
    counts: dict[str, int] = {}
    for variant in VARIANTS:
        out_dir = Path(run_dir) / f"entities_{variant}"
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for outcome in outcomes:
            rows = serialize(outcome, variant)
            total += len(rows)
            (out_dir / f"{outcome.document}.json").write_text(
                json.dumps(rows, ensure_ascii=False), encoding="utf-8"
            )
        counts[variant] = total
        log(f"variant {variant:16s} entities {total}")

    records = Path(run_dir) / "proposal-records.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "document": o.document,
                    "proposals": [p.as_dict() for p in o.pool.sorted()],
                    "entities": [
                        {
                            **e.as_dict(),
                            "assertions": list(o.assertions.get((e.start, e.end), ())),
                        }
                        for e in o.entities
                    ],
                    "counters": o.counters,
                },
                ensure_ascii=False, sort_keys=True,
            )
            + "\n"
            for o in outcomes
        ),
        encoding="utf-8",
    )
    return counts


def diagnostics(outcomes: list[DocumentOutcome], counts: dict[str, int]) -> dict[str, Any]:
    """Counts by proposal source, verifier action and final type."""
    totals: dict[str, int] = {}
    for outcome in outcomes:
        for key, value in outcome.counters.items():
            totals[key] = totals.get(key, 0) + value

    by_source = {
        "e3_only": 0, "qwen_only": 0, "alias_only": 0, "agreement": 0,
    }
    for outcome in outcomes:
        for entity in outcome.entities:
            sources = {s for s in entity.sources if s in (SOURCE_E3, SOURCE_QWEN, SOURCE_ALIAS)}
            if len(sources) > 1:
                by_source["agreement"] += 1
            elif sources == {SOURCE_E3}:
                by_source["e3_only"] += 1
            elif sources == {SOURCE_QWEN}:
                by_source["qwen_only"] += 1
            elif sources == {SOURCE_ALIAS}:
                by_source["alias_only"] += 1
    return {
        "documents": len(outcomes),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "final_entities_by_proposal_source": by_source,
        "variant_entity_counts": counts,
        "counters": dict(sorted(totals.items())),
    }


__all__ = [
    "VARIANTS",
    "VARIANT_BROAD",
    "VARIANT_CONTROL",
    "VARIANT_HIGH",
    "ContextualConfig",
    "DocumentOutcome",
    "Responder",
    "collect_proposals",
    "diagnostics",
    "process_document",
    "serialize",
    "variant_entities",
    "verify_pool",
    "write_variants",
]
