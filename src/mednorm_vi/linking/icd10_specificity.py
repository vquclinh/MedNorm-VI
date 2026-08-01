"""Hierarchy-aware ICD specificity arbitration (Audit 0070 §4-§6).

Audit 0069 repaired 1,342 damaged ICD titles and, in doing so, created a new failure mode.
Its three remaining top-1 losses were all one mechanism:

    `assess_specificity` measures a child by what it *adds* over its parent. Once repair gave
    B06 the title `Bệnh rubella [sởi Đức]`, its child B069 (`Bệnh rubella`) added nothing, so
    it earned no `supported_specific` credit and fell to `TIER_LEXICAL` - while the parent
    arrived by upward expansion and took `TIER_BROADER`, which sorts above lexical.

So a repaired parent whose title is a token-superset of its child's automatically demoted
that child. That is general: it fires for any family where the parent's name happens to be
the longer one, which repair made far more common.

The fix is to stop asking "what does the child add?" and start asking, of every member of a
hierarchy family, **"how much does this candidate assert that the text does not say?"**

    unsupported_extra   content tokens in the candidate's title that are absent from the
                        mention (and context, when available). Detail claimed but not stated.
    supported_added     content tokens the candidate adds over its family root that the text
                        *does* state. Detail claimed and stated.

Both directions matter and they resolve the two observed families in opposite ways with one
rule:

* B06 `Bệnh rubella [sởi Đức]` carries `sởi`/`đức`, which the mention does not - two
  unsupported tokens. B069 `Bệnh rubella` carries none. The **child** wins.
* Q01 `Dị tật thoát vị não` asserts nothing extra; Q018 adds a site qualifier the text does
  not state. The **parent** wins.

Arbitration is deliberately confined to *within* a hierarchy family. Choosing between two
unrelated concepts is a lexical question that Audit 0069's evidence tiers already answer;
this module only decides which member of a family represents it, and never moves a family
past another. That containment is why it cannot reintroduce the `sởi`/`sỏi` false friend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..kb.indexing import evidence as ev
from ..kb.indexing.retrieval import LocalIndex
from .icd10_hierarchy import canonical_name, content_tokens

#: H0 - Audit-0069 behaviour, no arbitration. The control.
POLICY_NONE = "H0_none"
#: H1 - intervene only to undo demonstrable over-assertion: a representative that claims
#: detail the text does not state loses to a family member that claims none. Anything else
#: keeps Audit-0069 order.
POLICY_CONSERVATIVE = "H1_conservative_unsupported_only"
#: H2 - full lexicographic family arbitration over the contract below.
POLICY_HIERARCHY_AWARE = "H2_hierarchy_aware_supported_specificity"

POLICIES: tuple[str, ...] = (POLICY_NONE, POLICY_CONSERVATIVE, POLICY_HIERARCHY_AWARE)

#: An ICD family is a three-character category and everything beneath it. Family membership
#: is structural, taken from the code itself, so it needs no graph lookup and cannot drift.
FAMILY_ROOT_LENGTH = 3


def family_root(code: str) -> str:
    return (code or "")[:FAMILY_ROOT_LENGTH]


@dataclass(frozen=True, slots=True)
class HierarchyFeatures:
    """Deterministic features for one candidate within its hierarchy family (§4).

    Every field is derived from the governed KB, the mention and (when supplied) the bounded
    context. Nothing here encodes a medical fact that the KB does not already state.
    """

    code: str
    family: str
    evidence_tier: str
    evidence_tier_index: int
    depth: int
    is_family_root: bool
    is_parent_of_representative: bool
    is_child_of_representative: bool
    parent_child_distance: int
    unsupported_extra: int
    supported_added: int
    mention_exact_to_title: bool
    mention_exact_to_alias: bool
    semantic_synonym_match: bool
    child_is_unspecified: bool
    child_is_other: bool
    sibling_competition_count: int
    lexical_score: float
    unsupported_tokens: tuple[str, ...] = field(default_factory=tuple)
    supported_tokens: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "family": self.family,
            "evidence_tier": self.evidence_tier,
            "depth": self.depth,
            "is_family_root": self.is_family_root,
            "parent_child_distance": self.parent_child_distance,
            "unsupported_extra": self.unsupported_extra,
            "supported_added": self.supported_added,
            "mention_exact_to_title": self.mention_exact_to_title,
            "mention_exact_to_alias": self.mention_exact_to_alias,
            "semantic_synonym_match": self.semantic_synonym_match,
            "child_is_unspecified": self.child_is_unspecified,
            "child_is_other": self.child_is_other,
            "sibling_competition_count": self.sibling_competition_count,
            "unsupported_tokens": list(self.unsupported_tokens),
            "supported_tokens": list(self.supported_tokens),
        }


def _last_character(code: str) -> str:
    return code[FAMILY_ROOT_LENGTH:][:1] if len(code) > FAMILY_ROOT_LENGTH else ""


def compute_features(
    index: LocalIndex,
    code: str,
    *,
    channels: tuple[str, ...],
    lexical_score: float,
    mention_text: str,
    context_text: str = "",
    sibling_competition: int = 0,
) -> HierarchyFeatures:
    """Features for one candidate. ``context_text`` is optional bounded context.

    Context participates only in *support*: it can justify a qualifier the mention omits, but
    it can never invent one. That asymmetry is deliberate - widening what counts as stated is
    safe, widening what counts as asserted is not.
    """
    root = family_root(code)
    title = canonical_name(index, code)
    title_tokens = content_tokens(title)
    # Always the root's own tokens - including when this candidate *is* the root, where the
    # subtraction must yield nothing. Zeroing it there would credit the category with its
    # whole title as "added detail" and let it out-argue every child it contains.
    root_tokens = content_tokens(canonical_name(index, root))
    stated = content_tokens(mention_text) | content_tokens(context_text)

    unsupported = title_tokens - stated
    added = title_tokens - root_tokens
    supported = added & stated
    tier = ev.tier_of(channels)
    suffix = _last_character(code)

    return HierarchyFeatures(
        code=code,
        family=root,
        evidence_tier=tier,
        evidence_tier_index=ev.TIER_ORDER.index(tier) if tier else ev.UNTIERED_RANK,
        depth=max(0, len(code) - FAMILY_ROOT_LENGTH),
        is_family_root=code == root,
        is_parent_of_representative=False,
        is_child_of_representative=False,
        parent_child_distance=max(0, len(code) - FAMILY_ROOT_LENGTH),
        unsupported_extra=len(unsupported),
        supported_added=len(supported),
        mention_exact_to_title="exact_canonical" in channels,
        mention_exact_to_alias="exact_alias" in channels,
        semantic_synonym_match="exact_synonym" in channels,
        child_is_unspecified=suffix == "9",
        child_is_other=suffix == "8",
        sibling_competition_count=sibling_competition,
        lexical_score=lexical_score,
        unsupported_tokens=tuple(sorted(unsupported)),
        supported_tokens=tuple(sorted(supported)),
    )


def representative_key(features: HierarchyFeatures) -> tuple[int, int, int, int, int, str]:
    """Which family member should represent the family. Ascending; first element wins.

    The ordering *is* the contract - there are no additive bonuses, so no amount of weaker
    evidence can accumulate past a stronger criterion:

    1. ``evidence_tier_index`` - Audit 0069's A-F, unchanged and dominant. A weak fuzzy child
       can never displace an exact parent, whatever its hierarchy position.
    2. ``unsupported_extra`` - **CASE B and CASE E.** A candidate asserting detail the text
       does not state loses. This is what stops `.8` winning on lexical similarity and what
       demotes a repaired parent carrying extra words the mention lacks.
    3. ``-supported_added`` - **CASE A.** When the text *does* state the qualifier, the child
       that carries it wins. Detail that is both claimed and stated is the strongest reason
       to be specific.
    4. ``not is_family_root`` - **CASE C and CASE D.** When a parent and its unspecified child
       are indistinguishable on the evidence, the category represents the family, and when
       several children remain equally plausible none of them is picked arbitrarily. Both
       follow from one rule: ties resolve upward, never sideways.
    5. ``depth`` then ``code`` - deterministic, so a repeat run cannot reorder.
    """
    return (
        features.evidence_tier_index,
        features.unsupported_extra,
        -features.supported_added,
        0 if features.is_family_root else 1,
        features.depth,
        features.code,
    )


def arbitrate_order(
    ordered_codes: list[str],
    features: dict[str, HierarchyFeatures],
    *,
    policy: str = POLICY_HIERARCHY_AWARE,
) -> list[str]:
    """Reorder candidates so each family is represented by its best member.

    Families never move relative to each other: a family keeps the best position any of its
    members already held, and only *which* member sits there can change. Confining the policy
    this way means it cannot promote an unrelated concept, so the Audit-0069 tier results -
    including `sởi` -> B05 - are structurally preserved.
    """
    if policy == POLICY_NONE or not ordered_codes:
        return list(ordered_codes)

    families: dict[str, list[str]] = {}
    for code in ordered_codes:
        families.setdefault(family_root(code), []).append(code)

    promoted: dict[str, str] = {}
    for root, members in families.items():
        if len(members) < 2:
            continue
        known = [m for m in members if m in features]
        if len(known) < 2:
            continue
        current = known[0]
        best = min(known, key=lambda m: representative_key(features[m]))
        if best == current:
            continue
        if policy == POLICY_CONSERVATIVE:
            # Intervene only on demonstrable over-assertion: the incumbent claims detail the
            # text does not state and the challenger claims none, at no worse evidence tier.
            incumbent, challenger = features[current], features[best]
            if not (
                incumbent.unsupported_extra > 0
                and challenger.unsupported_extra == 0
                and challenger.evidence_tier_index <= incumbent.evidence_tier_index
            ):
                continue
        promoted[root] = best

    if not promoted:
        return list(ordered_codes)

    result: list[str] = []
    emitted: set[str] = set()
    for code in ordered_codes:
        root = family_root(code)
        winner = promoted.get(root)
        if winner is None:
            result.append(code)
            continue
        if code == winner:
            # Its slot was already taken by the promotion below; skip the duplicate.
            if winner not in emitted:
                result.append(winner)
                emitted.add(winner)
            continue
        if winner not in emitted:
            result.append(winner)
            emitted.add(winner)
        result.append(code)
    return result


def policy_document(policy: str) -> dict[str, Any]:
    """The arbitration contract as data, for the run manifest."""
    return {
        "policy_id": policy,
        "audit": "0070",
        "scope": (
            "within an ICD hierarchy family only; families never reorder relative to each other"
        ),
        "representative_key": [
            "evidence_tier_index",
            "unsupported_extra",
            "-supported_added",
            "not_is_family_root",
            "depth",
            "code",
        ],
        "cases": {
            "A_context_supports_child": "child wins via -supported_added",
            "B_child_adds_unsupported_qualifier": "parent wins via unsupported_extra",
            "C_child_is_unspecified_realisation": "tie resolves upward to the category",
            "D_multiple_plausible_children": "no child chosen arbitrarily; category represents",
            "E_other_and_unspecified": (
                ".8 loses on unsupported_extra because its site/qualifier tokens are unstated; "
                ".9 adds only stopwords so it ties and loses to the category by rule C"
            ),
        },
        "invariants": [
            "evidence tier dominates: a weak fuzzy child cannot defeat an exact parent",
            "no additive bonuses; ordering is lexicographic",
            "arbitration never moves a family past another family",
            "deterministic: total order, ties broken by code",
        ],
    }


__all__ = [
    "FAMILY_ROOT_LENGTH",
    "POLICIES",
    "POLICY_CONSERVATIVE",
    "POLICY_HIERARCHY_AWARE",
    "POLICY_NONE",
    "HierarchyFeatures",
    "arbitrate_order",
    "compute_features",
    "family_root",
    "policy_document",
    "representative_key",
]
