"""Runtime search indices derived from the competition candidate v3 snapshot (§10).

The runtime consumes a compact ``index.json``, not the v3 CSV tables, so activating
v3 means building an index from it. This module does that, and it exists as a separate
step from :mod:`.build` because the two answer different questions: the snapshot is the
audited evidence, the index is a derived lookup structure that can be regenerated from
it at any time.

**The closure population is the whole difficulty.** Graph traversal needs the 129,520
closure-only concepts to be present — that is why they were added — but the runtime
must never emit one. Presence and emittability are therefore separated inside the
index itself:

    records          searchable + closure-only, so traversal can resolve a neighbour
    postings         searchable surfaces only, so retrieval can never reach a
                     closure node in the first place
    metadata.membership   carried per record, so the linker and L9 can refuse one
                          that traversal reached

Retrieval-side exclusion alone would not be enough: traversal walks the adjacency,
not the postings, so it reaches closure nodes by construction. The membership flag is
what stops the walk from turning into a candidate — see
``linking/rxnorm.py``'s ``DROP_CLOSURE_ONLY`` and ``validator/kb_membership.py``.

The ``graph`` field is written as the same **symmetric unlabeled adjacency** the legacy
index uses. That is deliberate: it keeps ``rxnorm_graph``'s TTY-role traversal working
unchanged, so activating v3 needs no redesign of the canonical runner. The directed,
labelled relations remain in ``rxnorm_competition_relations_v1.csv``, which is what a
future relation-aware walk would read.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..indexing.builders import BUILDER_VERSION, _add_alias, _freeze_postings, _index_hash
from ..indexing.models import IndexMetadata, IndexRecord, SearchIndex
from .rxnorm_view import CONCEPT_CLOSURE_ONLY, CONCEPT_SEARCHABLE

COMPETITION_INDEX_BUILDER_VERSION = "competition-index-v1"

# Alias surfaces per concept are bounded so one hub concept cannot dominate the
# postings. A bound, not a selection rule: the canonical name is always indexed.
MAX_ALIASES_PER_CONCEPT = 32


def _write_index(index: SearchIndex, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metadata": {
            "index_id": index.metadata.index_id,
            "index_type": index.metadata.index_type,
            "source_snapshot_id": index.metadata.source_snapshot_id,
            "source_hash": index.metadata.source_hash,
            "config_hash": index.metadata.config_hash,
            "builder_version": index.metadata.builder_version,
            "deterministic_index_hash": index.metadata.deterministic_index_hash,
            "record_count": index.metadata.record_count,
            "concept_count": index.metadata.concept_count,
            "parameters": index.metadata.parameters,
        },
        "records": [
            {
                "concept_id": r.concept_id,
                "canonical_name": r.canonical_name,
                "aliases": list(r.aliases),
                "metadata": r.metadata,
            }
            for r in index.records
        ],
        "exact": {k: list(v) for k, v in index.exact.items()},
        "exact_ascii": {k: list(v) for k, v in index.exact_ascii.items()},
        "ngrams": {k: list(v) for k, v in index.ngrams.items()},
        "sparse_terms": {k: list(v) for k, v in index.sparse_terms.items()},
        "graph": {k: list(v) for k, v in index.graph.items()},
    }
    path = output_dir / "index.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _numeric_key(value: str) -> tuple[int, int, str]:
    return (0, int(value), "") if value.isdigit() else (1, 0, value)


def build_competition_icd_index(
    snapshot_dir: str | Path, output_dir: str | Path, *, snapshot_id: str
) -> IndexMetadata:
    """Build the ICD runtime index from the v3 competition view.

    Every structurally valid concept is indexed, including the 3,470 with suspect
    names: ``metadata.name_quality`` carries the penalty band so ranking can use it,
    which is the competition policy's whole point — penalise, never exclude.
    """
    root = Path(snapshot_dir)
    exact: dict[str, set[str]] = defaultdict(set)
    exact_ascii: dict[str, set[str]] = defaultdict(set)
    ngrams: dict[str, set[str]] = defaultdict(set)
    sparse: dict[str, set[str]] = defaultdict(set)
    graph: dict[str, set[str]] = defaultdict(set)
    records: list[IndexRecord] = []

    searchable: set[str] = set()
    with (root / "icd10_competition_concepts_v1.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("searchable") != "true":
                continue
            code = row["code"]
            searchable.add(code)
            name = row.get("source_name", "")
            for alias in (name, row.get("display_code", ""), code):
                if alias:
                    _add_alias(
                        code,
                        alias,
                        exact=exact,
                        exact_ascii=exact_ascii,
                        ngrams=ngrams,
                        sparse_terms=sparse,
                    )
            records.append(
                IndexRecord(
                    concept_id=code,
                    canonical_name=name,
                    aliases=(),
                    metadata={
                        "dotted_code": row.get("display_code", ""),
                        "name_quality": row.get("name_quality", ""),
                        "quality_flags": row.get("quality_flags", ""),
                        "membership": CONCEPT_SEARCHABLE,
                        "runtime_role": row.get("runtime_role", ""),
                        "specificity": str(max(len(code) - 3, 0)),
                    },
                )
            )

    with (root / "icd10_advisory_hierarchy_v1.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            parent, child = row.get("parent_code", ""), row.get("child_code", "")
            if parent in searchable and child in searchable:
                # Symmetric, exactly like the legacy index, so the hierarchy linker
                # needs no change. The edge's advisory status lives in the CSV.
                graph[parent].add(child)
                graph[child].add(parent)

    records.sort(key=lambda r: r.concept_id)
    return _finalize(
        records,
        exact,
        exact_ascii,
        ngrams,
        sparse,
        graph,
        index_type="icd10_vi",
        snapshot_id=snapshot_id,
        output_dir=Path(output_dir),
        parameters={
            "ngram_n": 3,
            "normalization": "casefold+NFKC+accent_optional",
            "source_view": "competition-icd10-view-v1",
            "suspect_names_indexed": True,
            "hierarchy_evidence": "advisory_prefix_inference",
        },
    )


def build_competition_rxnorm_index(
    snapshot_dir: str | Path, output_dir: str | Path, *, snapshot_id: str
) -> IndexMetadata:
    """Build the RxNorm runtime index from the v3 competition view.

    Closure-only concepts enter ``records`` and ``graph`` but never the postings, and
    every record states its membership so a traversal result can be refused downstream.
    """
    root = Path(snapshot_dir)
    exact: dict[str, set[str]] = defaultdict(set)
    exact_ascii: dict[str, set[str]] = defaultdict(set)
    ngrams: dict[str, set[str]] = defaultdict(set)
    sparse: dict[str, set[str]] = defaultdict(set)
    graph: dict[str, set[str]] = defaultdict(set)

    membership: dict[str, str] = {}
    concept_meta: dict[str, dict[str, str]] = {}
    names: dict[str, str] = {}
    with (root / "rxnorm_competition_concepts_v1.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rxcui = row["rxcui"]
            membership[rxcui] = row.get("membership", "")
            names[rxcui] = row.get("preferred_name", "")
            ttys = [t for t in row.get("tty_values", "").split("|") if t]
            concept_meta[rxcui] = {
                # The runtime graph reads a single `tty`; the full observed collection
                # is preserved beside it rather than being thrown away.
                "tty": ttys[0] if len(ttys) == 1 else _primary_tty(ttys),
                "tty_values": row.get("tty_values", ""),
                "suppress": _primary_suppress(row.get("suppress_values", "")),
                "membership": row.get("membership", ""),
                "runtime_role": row.get("runtime_role", ""),
            }

    aliases_by_cui: dict[str, list[str]] = defaultdict(list)
    with (root / "rxnorm_governed_synonyms_v1.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rxcui = row.get("concept_id", "")
            # Only a searchable concept's surfaces enter the postings, so retrieval
            # cannot reach a closure node at all.
            if membership.get(rxcui) != CONCEPT_SEARCHABLE:
                continue
            surface = row.get("surface", "")
            if surface and len(aliases_by_cui[rxcui]) < MAX_ALIASES_PER_CONCEPT:
                aliases_by_cui[rxcui].append(surface)

    with (root / "rxnorm_competition_relations_v1.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            source, target = row.get("source_rxcui", ""), row.get("target_rxcui", "")
            if source and target:
                graph[source].add(target)
                graph[target].add(source)

    records: list[IndexRecord] = []
    for rxcui in sorted(membership, key=_numeric_key):
        surfaces = sorted(set(aliases_by_cui.get(rxcui, ())))
        name = names.get(rxcui, "") or (surfaces[0] if surfaces else "")
        if membership[rxcui] == CONCEPT_SEARCHABLE:
            for alias in {name, *surfaces} - {""}:
                _add_alias(
                    rxcui,
                    alias,
                    exact=exact,
                    exact_ascii=exact_ascii,
                    ngrams=ngrams,
                    sparse_terms=sparse,
                )
        records.append(
            IndexRecord(
                concept_id=rxcui,
                canonical_name=name,
                aliases=tuple(surfaces),
                metadata=concept_meta[rxcui],
            )
        )

    return _finalize(
        records,
        exact,
        exact_ascii,
        ngrams,
        sparse,
        graph,
        index_type="rxnorm",
        snapshot_id=snapshot_id,
        output_dir=Path(output_dir),
        parameters={
            "ngram_n": 3,
            "normalization": "casefold+NFKC+accent_optional",
            "source_view": "competition-rxnorm-view-v1",
            "searchable_concepts": sum(1 for m in membership.values() if m == CONCEPT_SEARCHABLE),
            "closure_only_concepts": sum(
                1 for m in membership.values() if m == CONCEPT_CLOSURE_ONLY
            ),
            "closure_only_in_postings": False,
            "graph_edges_are_labeled": False,
            "labeled_relations_artifact": "rxnorm_competition_relations_v1.csv",
        },
    )


def _primary_tty(ttys: Sequence[str]) -> str:
    """One representative TTY for a concept observed with several.

    Product term types outrank ingredient ones, which outrank string types, so a
    concept that is both ``IN`` and ``SU`` presents as ``IN`` to the TTY-role walk
    rather than as whichever string happened to sort first.
    """
    for preferred in ("SCD", "SBD", "SCDC", "SBDC", "GPCK", "BPCK", "MIN", "PIN", "IN", "BN"):
        if preferred in ttys:
            return preferred
    return sorted(ttys)[0] if ttys else ""


def _primary_suppress(values: str) -> str:
    """``N`` only when the concept has an unsuppressed atom."""
    observed = {v for v in values.split("|") if v}
    return "N" if ("N" in observed or not observed) else sorted(observed)[0]


def _finalize(
    records: list[IndexRecord],
    exact: dict[str, set[str]],
    exact_ascii: dict[str, set[str]],
    ngrams: dict[str, set[str]],
    sparse: dict[str, set[str]],
    graph: dict[str, set[str]],
    *,
    index_type: str,
    snapshot_id: str,
    output_dir: Path,
    parameters: dict[str, Any],
) -> IndexMetadata:
    payload = {
        "records": [(r.concept_id, r.canonical_name, r.aliases, r.metadata) for r in records],
        "exact": {k: sorted(v) for k, v in _freeze_postings(exact).items()},
        "exact_ascii": {k: sorted(v) for k, v in _freeze_postings(exact_ascii).items()},
        "ngrams": {k: sorted(v) for k, v in _freeze_postings(ngrams).items()},
        "sparse_terms": {k: sorted(v) for k, v in _freeze_postings(sparse).items()},
        "graph": {k: sorted(v) for k, v in _freeze_postings(graph).items()},
    }
    deterministic_hash = _index_hash(payload)
    metadata = IndexMetadata(
        index_id=f"{index_type}-competition-index-{deterministic_hash[:16]}",
        index_type=index_type,
        source_snapshot_id=snapshot_id,
        source_hash=deterministic_hash,
        config_hash=deterministic_hash[:32],
        builder_version=f"{BUILDER_VERSION}+{COMPETITION_INDEX_BUILDER_VERSION}",
        deterministic_index_hash=deterministic_hash,
        record_count=len(records),
        concept_count=len({r.concept_id for r in records}),
        parameters=parameters,
    )
    _write_index(
        SearchIndex(
            metadata=metadata,
            records=tuple(records),
            exact=_freeze_postings(exact),
            exact_ascii=_freeze_postings(exact_ascii),
            ngrams=_freeze_postings(ngrams),
            sparse_terms=_freeze_postings(sparse),
            graph=_freeze_postings(graph),
        ),
        output_dir,
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build v3-derived runtime indices.")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--icd-output-dir", required=True)
    parser.add_argument("--rxnorm-output-dir", required=True)
    parser.add_argument("--snapshot-id", required=True)
    args = parser.parse_args(argv)
    icd = build_competition_icd_index(
        args.snapshot_dir, args.icd_output_dir, snapshot_id=args.snapshot_id
    )
    rxnorm = build_competition_rxnorm_index(
        args.snapshot_dir, args.rxnorm_output_dir, snapshot_id=args.snapshot_id
    )
    json.dump(
        {
            "icd10": {
                "index_id": icd.index_id,
                "records": icd.record_count,
                "deterministic_index_hash": icd.deterministic_index_hash,
                "parameters": icd.parameters,
            },
            "rxnorm": {
                "index_id": rxnorm.index_id,
                "records": rxnorm.record_count,
                "deterministic_index_hash": rxnorm.deterministic_index_hash,
                "parameters": rxnorm.parameters,
            },
        },
        sys.stdout,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPETITION_INDEX_BUILDER_VERSION",
    "MAX_ALIASES_PER_CONCEPT",
    "build_competition_icd_index",
    "build_competition_rxnorm_index",
    "main",
]
