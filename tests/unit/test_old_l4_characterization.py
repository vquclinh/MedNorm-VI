"""Characterization of the obsolete non-canonical L4 (Audit 0054 §10, step 1).

Milestone 2B §10 requires five steps before ``resolution/resolver.py`` can be deleted:
characterize, migrate, prove equivalence, document, then remove. **This file is step 1
only.** It pins the behaviour that is unique to the old resolver so a future migration
has something to migrate *against*, rather than a reading of the source.

Why the deletion did not happen in Audit 0054, stated plainly: the old resolver is
imported by three live surfaces —

```text
src/mednorm_vi/resolution/__init__.py     re-exports ResolverConfig, resolve
src/mednorm_vi/phase1c_foundation/cli.py  a separate CLI that runs it
src/mednorm_vi/phase1c_foundation/doctor.py  validates its config loads
tests/unit/test_resolution.py             its existing test suite
```

Deleting the module means migrating a whole CLI surface as well, and doing that in the
same turn as six other stages would have meant proving none of it. The characterization
below is the honest half of the work: it is now recorded what would be lost.

**The unique behaviour is a configurable per-type boundary policy.** The canonical L4
(``resolution/canonical.py`` over ``resolver_v1``) shapes boundaries by evidence-weighted
trim/expand actions and has no equivalent of these policy switches:

```text
ResolverConfig.medication_boundary    "full" | "name_only" | "name_strength"
ResolverConfig.test_result_boundary   "value_only" | "value_unit" | "full"
ResolverConfig.abstain_on_conflict    bool
```

Everything else the old resolver does — grouping, type assignment, overlap resolution,
`has_result` retention, scoring — the canonical path also does, and does with more
evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mednorm_vi.mention_factory.models import RelationProposal, SpanProposal
from mednorm_vi.resolution.resolver import ResolverConfig, resolve

REPO = Path(__file__).resolve().parents[2]
OLD_RESOLVER = REPO / "src" / "mednorm_vi" / "resolution" / "resolver.py"


def _proposal(
    proposal_id: str,
    start: int,
    end: int,
    text: str,
    *,
    types: tuple[str, ...] = ("THUỐC",),
    specialist: str = "medication",
    boundary_group_id: str = "bg1",
    matched_rule: str = "med_grammar:full",
    local_score: float = 0.9,
) -> SpanProposal:
    """One E1/E2-shaped proposal.

    ``matched_rule`` matters: ``features.boundary_kind`` derives the boundary kind from
    it, and the old resolver's policy switches select by kind. A proposal without a rule
    string classifies as `single`, which is why every proposal here carries one.
    """
    return SpanProposal(
        proposal_id=proposal_id, document_id="d1", start=start, end=end, text=text,
        proposed_types=types, source_specialist=specialist,
        source_node_id="n1", source_routes=("C1",), local_score=local_score,
        matched_rule=matched_rule, boundary_group_id=boundary_group_id)


MEDICATION_TEXT = "aspirin 81 mg po"


def _medication_group() -> list[SpanProposal]:
    """Three competing boundaries for one logical medication mention."""
    return [
        _proposal("p-name", 0, 7, "aspirin", matched_rule="med_grammar:name_only"),
        _proposal("p-strength", 0, 13, "aspirin 81 mg",
                  matched_rule="med_grammar:name_strength"),
        _proposal("p-full", 0, 16, MEDICATION_TEXT, matched_rule="med_grammar:full"),
    ]


# ---------------------------------------------------------------------------
# Unique behaviour 1: configurable medication boundary policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "expected_text"),
    [
        ("full", MEDICATION_TEXT),
        ("name_only", "aspirin"),
        ("name_strength", "aspirin 81 mg"),
    ],
)
def test_medication_boundary_policy_selects_a_different_span(
    policy: str, expected_text: str
) -> None:
    """The policy switch is the old resolver's one irreplaceable behaviour.

    The canonical L4 shapes boundaries from evidence and has no equivalent knob, so a
    migration must either reproduce these three outcomes or record a deliberate change.
    """
    result = resolve(
        "d1", MEDICATION_TEXT, _medication_group(), [],
        ResolverConfig(medication_boundary=policy))
    accepted = result.accepted()
    assert len(accepted) == 1
    assert accepted[0].text == expected_text
    assert accepted[0].boundary_evidence.policy == policy


def test_the_chosen_boundary_records_its_policy_and_alternatives() -> None:
    result = resolve(
        "d1", MEDICATION_TEXT, _medication_group(), [],
        ResolverConfig(medication_boundary="name_only"))
    hypothesis = result.accepted()[0]
    assert hypothesis.boundary_evidence.policy == "name_only"
    assert {a.proposal_id for a in hypothesis.retained_alternatives} == {
        "p-strength", "p-full"}
    assert hypothesis.source_proposal_ids == ("p-name", "p-strength", "p-full")


# ---------------------------------------------------------------------------
# Unique behaviour 2: configurable test-result boundary policy
# ---------------------------------------------------------------------------


LAB_TEXT = "WBC: 14.43 K/uL"


@pytest.mark.parametrize(
    ("policy", "expected_text"),
    [
        ("value_only", "14.43"),
        ("value_unit", "14.43 K/uL"),
    ],
)
def test_test_result_boundary_policy_selects_a_different_span(
    policy: str, expected_text: str
) -> None:
    proposals = [
        _proposal("r-value", 5, 10, "14.43", types=("KẾT_QUẢ_XÉT_NGHIỆM",),
                  specialist="laboratory", boundary_group_id="bg-r",
                  matched_rule="lab:test_result:value_only:r1"),
        _proposal("r-unit", 5, 15, "14.43 K/uL", types=("KẾT_QUẢ_XÉT_NGHIỆM",),
                  specialist="laboratory", boundary_group_id="bg-r",
                  matched_rule="lab:test_result:value_unit:r1"),
    ]
    result = resolve(
        "d1", LAB_TEXT, proposals, [], ResolverConfig(test_result_boundary=policy))
    accepted = result.accepted()
    assert len(accepted) == 1
    assert accepted[0].text == expected_text


# ---------------------------------------------------------------------------
# Unique behaviour 3: has_result retention without requiring pairing
# ---------------------------------------------------------------------------


def test_pair_group_evidence_is_retained_on_both_endpoints() -> None:
    proposals = [
        _proposal("n1", 0, 3, "WBC", types=("TÊN_XÉT_NGHIỆM",),
                  specialist="laboratory", boundary_group_id="bg-n",
                  matched_rule="lab:test_name:n1"),
        _proposal("r1", 5, 10, "14.43", types=("KẾT_QUẢ_XÉT_NGHIỆM",),
                  specialist="laboratory", boundary_group_id="bg-r",
                  matched_rule="lab:test_result:value_only:r1"),
    ]
    relations = [RelationProposal(
        relation_id="rel1", document_id="d1", relation_type="HAS_RESULT",
        source_proposal_id="n1", target_proposal_id="r1", score=0.9,
        pairing_cost=0.1, pair_group_id="g1", is_primary=True)]
    result = resolve("d1", LAB_TEXT, proposals, relations, ResolverConfig())
    for hypothesis in result.accepted():
        assert hypothesis.has_result_pair_group_ids == ("g1",)


# ---------------------------------------------------------------------------
# Unique behaviour 4: types outside med/lab are UNRESOLVED, with a warning
# ---------------------------------------------------------------------------


def test_an_unsupported_type_is_unresolved_and_warned() -> None:
    proposals = [_proposal(
        "d1p", 0, 9, "viem phoi", types=("CHẨN_ĐOÁN",), specialist="diagnosis",
        boundary_group_id="bg-d", matched_rule="")]
    result = resolve("d1", "viem phoi", proposals, [], ResolverConfig())
    assert result.accepted() == ()
    assert len(result.unresolved()) == 1
    assert any(w.startswith("unresolvable_type:") for w in result.warnings)


# ---------------------------------------------------------------------------
# Unique behaviour 5: abstain_on_conflict
# ---------------------------------------------------------------------------


def test_abstain_on_conflict_is_a_distinct_configuration() -> None:
    """Pinned as a *configuration surface*, not as an outcome.

    The flag reaches `resolve_overlaps`; this test records that both settings load and
    resolve without error, which is what a migration has to preserve. It deliberately
    does not assert a specific overlap outcome, because doing so would pin a behaviour
    the canonical L4 already implements differently and better (§7.3 competition).
    """
    proposals = _medication_group()
    for flag in (False, True):
        result = resolve(
            "d1", MEDICATION_TEXT, proposals, [],
            ResolverConfig(abstain_on_conflict=flag))
        assert result.config_version == "resolver-v1"
        assert len(result.hypotheses) >= 1


def test_config_loads_from_yaml_and_hashes_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "resolver.yaml"
    path.write_text(
        "config_version: resolver-v1\nmedication_boundary: name_only\n"
        "test_result_boundary: value_unit\nabstain_on_conflict: true\n",
        encoding="utf-8")
    first = ResolverConfig.load(path)
    second = ResolverConfig.load(path)
    assert first == second
    assert first.medication_boundary == "name_only"
    assert first.test_result_boundary == "value_unit"
    assert first.abstain_on_conflict is True
    assert len(first.config_hash) == 64


# ---------------------------------------------------------------------------
# Deletion readiness: recorded, not yet satisfied
# ---------------------------------------------------------------------------


def test_the_old_resolver_is_still_present_and_this_is_deliberate() -> None:
    """Guards against an undocumented deletion.

    When the migration lands, this test is the one to remove — and removing it should
    require reading Audit 0054 §12, which lists what has to be true first.
    """
    assert OLD_RESOLVER.is_file(), (
        "resolution/resolver.py was deleted; if the migration is complete, remove this "
        "characterization module and update ACTIVE_RUNTIME_MANIFEST.md in the same "
        "change")


def test_the_old_resolver_is_off_the_canonical_path() -> None:
    """It may exist; it may not be reachable from the runner."""
    import ast
    import inspect

    from mednorm_vi.inference import pipeline

    tree = ast.parse(inspect.getsource(pipeline))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    assert not any(m.endswith("resolution.resolver") or m == "..resolution.resolver"
                   for m in modules), "the canonical runner must not import the old L4"
    assert any("resolution.canonical" in m for m in modules), (
        "the canonical runner must import the canonical L4")


def test_the_known_importers_of_the_old_resolver_are_the_documented_ones() -> None:
    """If a new importer appears, the migration got harder and the audit is stale."""
    import subprocess

    output = subprocess.run(
        ["git", "grep", "-l", "-E", r"from \.resolver import|resolution\.resolver import"],
        cwd=REPO, capture_output=True, text=True, check=False).stdout
    importers = {line.strip() for line in output.splitlines() if line.strip()}
    expected = {
        "src/mednorm_vi/resolution/__init__.py",
    }
    unexpected = importers - expected
    assert not unexpected, (
        f"new importer(s) of the obsolete L4: {sorted(unexpected)}; Audit 0054 §11 "
        "lists the ones that existed when the inventory was taken")
