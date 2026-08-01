"""Evidence tiers for candidate ranking (Audit 0069 §4).

Audit 0068 added 29,586 ICD aliases. Coverage rose - and top-1 accuracy fell, because the
retrieval score was a single additive float: exact matches contributed 100, accent-stripped
exact 80, and every shared trigram another 1. With enough aliases in the index a concept can
therefore accumulate more than 100 points of trigram noise and outrank a concept the query
matched *exactly*. That is not a tuning problem; a sum cannot express "no amount of fuzzy
evidence beats one exact match".

So ranking is lexicographic over an explicit ordered tuple instead. The tier decides first
and nothing below it can overturn that decision:

    Tier A  accent-sensitive exact match on the canonical title (or the code itself)
    Tier B  accent-sensitive exact match on a governed alias
    Tier C  governed semantic synonym-rule match
    Tier D  accent-sensitive sparse / character-n-gram evidence
    Tier E  accent-stripped exact match on a canonical title or alias
    Tier F  accent-stripped fuzzy evidence

The accent boundary between D and E is the point of the whole ordering. `sởi` (measles) and
`sỏi` (calculus) collapse to the same string once accents are stripped, so accent-stripped
evidence has to sit *below* every accent-preserving channel. It stays in the index because
it recovers concepts for users who type without diacritics - it just can no longer outrank
an accent-exact match.
"""

from __future__ import annotations

from typing import Any

TIER_A_EXACT_CANONICAL = "A"
TIER_B_EXACT_ALIAS = "B"
TIER_C_SEMANTIC_SYNONYM = "C"
TIER_D_ACCENT_FUZZY = "D"
TIER_E_UNACCENTED_EXACT = "E"
TIER_F_UNACCENTED_FUZZY = "F"

#: Strongest first. Position in this tuple *is* the ranking authority.
TIER_ORDER: tuple[str, ...] = (
    TIER_A_EXACT_CANONICAL,
    TIER_B_EXACT_ALIAS,
    TIER_C_SEMANTIC_SYNONYM,
    TIER_D_ACCENT_FUZZY,
    TIER_E_UNACCENTED_EXACT,
    TIER_F_UNACCENTED_FUZZY,
)

#: Retrieval channel -> tier. A channel absent here contributes no tier evidence at all.
CHANNEL_TIERS: dict[str, str] = {
    "exact_canonical": TIER_A_EXACT_CANONICAL,
    "exact": TIER_A_EXACT_CANONICAL,
    "exact_alias": TIER_B_EXACT_ALIAS,
    "exact_synonym": TIER_C_SEMANTIC_SYNONYM,
    "sparse_accent": TIER_D_ACCENT_FUZZY,
    "char_ngram_accent": TIER_D_ACCENT_FUZZY,
    "exact_ascii": TIER_E_UNACCENTED_EXACT,
    "sparse": TIER_F_UNACCENTED_FUZZY,
    "char_ngram": TIER_F_UNACCENTED_FUZZY,
}

#: Channels that matched a whole normalized string rather than a fragment of one.
EXACT_CHANNELS: frozenset[str] = frozenset(
    {"exact", "exact_canonical", "exact_alias", "exact_synonym", "exact_ascii"}
)

#: Channels carrying governed alias evidence, counted as semantic support.
ALIAS_CHANNELS: frozenset[str] = frozenset({"exact_alias", "exact_synonym"})

#: Rank assigned when an index carries no tier postings (competition-v3 and RxNorm). It sorts
#: below every real tier, so those indexes keep their pre-Audit-0069 score ordering exactly.
UNTIERED_RANK = len(TIER_ORDER) + 1


def tier_of(channels: tuple[str, ...] | frozenset[str]) -> str:
    """The strongest tier among ``channels``, or `""` when none carries tier evidence."""
    best = ""
    for channel in channels:
        tier = CHANNEL_TIERS.get(channel)
        if tier is None:
            continue
        if not best or TIER_ORDER.index(tier) < TIER_ORDER.index(best):
            best = tier
    return best


def rank_key(
    *,
    concept_id: str,
    channels: tuple[str, ...],
    lexical_score: float,
    hierarchy_signal: int = 0,
    tiered: bool = True,
) -> tuple[int, int, int, float, int, str]:
    """The ordered tuple a candidate is ranked by. Ascending: first element wins.

    Every component is negated where "more is better" so that one plain ascending sort
    expresses the whole contract, and the last element is the concept id, which makes the
    order total and therefore reproducible.

    * ``evidence_tier``          - A…F above; the only thing that can move a candidate past
                                   another that matched at a stronger tier.
    * ``exactness``              - a whole-string match beats a fragment match inside a tier.
    * ``semantic_alias_support`` - how many governed alias channels fired.
    * ``lexical_score``          - the accumulated fuzzy score, now a tie-breaker only.
    * ``hierarchy_signal``       - concepts whose ICD neighbourhood also matched.
    * ``deterministic_code``     - final tie-break, so equal evidence never reorders.
    """
    tier = tier_of(channels) if tiered else ""
    tier_index = TIER_ORDER.index(tier) if tier else UNTIERED_RANK
    exactness = 1 if any(c in EXACT_CHANNELS for c in channels) else 0
    alias_support = sum(1 for c in channels if c in ALIAS_CHANNELS)
    return (
        tier_index,
        -exactness,
        -alias_support,
        -float(lexical_score),
        -int(hierarchy_signal),
        concept_id,
    )


def policy_document() -> dict[str, Any]:
    """The scoring contract as data, so the index can ship the policy it was built for."""
    return {
        "policy_id": "icd10-evidence-tier-v1",
        "audit": "0069",
        "ranking": "lexicographic over an ordered tuple; no additive score constants",
        "rank_tuple": [
            "evidence_tier",
            "exactness",
            "semantic_alias_support",
            "lexical_score",
            "hierarchy_signal",
            "deterministic_code_tiebreak",
        ],
        "tiers": [
            {"tier": "A", "name": "accent_sensitive_exact_canonical"},
            {"tier": "B", "name": "accent_sensitive_exact_governed_alias"},
            {"tier": "C", "name": "governed_semantic_synonym_rule"},
            {"tier": "D", "name": "accent_sensitive_sparse_and_char_ngram"},
            {"tier": "E", "name": "accent_stripped_exact_canonical_or_alias"},
            {"tier": "F", "name": "accent_stripped_fuzzy"},
        ],
        "channel_tiers": dict(sorted(CHANNEL_TIERS.items())),
        "invariant": (
            "A lower tier can never outrank a higher tier, whatever fuzzy score it "
            "accumulates. Accent-stripped evidence (E, F) always sorts below every "
            "accent-preserving channel (A-D), so a one-diacritic false friend such as "
            "sỏi cannot displace an accent-exact match such as sởi."
        ),
        "untiered_indexes": (
            "An index without tier postings ranks at UNTIERED_RANK and keeps its previous "
            "score ordering, so competition-v3 and RxNorm are bit-for-bit unaffected."
        ),
    }


__all__ = [
    "ALIAS_CHANNELS",
    "CHANNEL_TIERS",
    "EXACT_CHANNELS",
    "TIER_A_EXACT_CANONICAL",
    "TIER_B_EXACT_ALIAS",
    "TIER_C_SEMANTIC_SYNONYM",
    "TIER_D_ACCENT_FUZZY",
    "TIER_E_UNACCENTED_EXACT",
    "TIER_F_UNACCENTED_FUZZY",
    "TIER_ORDER",
    "UNTIERED_RANK",
    "policy_document",
    "rank_key",
    "tier_of",
]
