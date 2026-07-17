"""Build deterministic local search indexes for ICD-10 and RxNorm."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..rxnorm.models import STRUCTURE_RELAS
from .models import IndexMetadata, IndexRecord, SearchIndex
from .normalization import char_ngrams, normalize_text, tokens

BUILDER_VERSION = "kb-index-v1"

_CONSO = {"RXCUI": 0, "LAT": 1, "ISPREF": 6, "RXAUI": 7, "SAB": 11,
          "TTY": 12, "CODE": 13, "STR": 14, "SUPPRESS": 16}
_REL = {"RXCUI1": 0, "REL": 3, "RXCUI2": 4, "RELA": 7, "SUPPRESS": 14}
_SAT = {"RXCUI": 0, "ATN": 8, "ATV": 9, "SUPPRESS": 11}
_ATTRIBUTES_TO_KEEP = frozenset(
    {
        "RXN_STRENGTH",
        "RXN_AVAILABLE_STRENGTH",
        "RXN_BN_CARDINALITY",
        "RXN_QUANTITY",
        "RXN_HUMAN_DRUG",
        "RXN_OBSOLETED",
    }
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        h.update(f"{rel}\t{_sha256_file(path)}\n".encode())
    return h.hexdigest()


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _freeze_postings(d: dict[str, set[str]]) -> dict[str, tuple[str, ...]]:
    return {k: tuple(sorted(v)) for k, v in sorted(d.items()) if v}


def _add_alias(
    concept_id: str,
    alias: str,
    *,
    exact: dict[str, set[str]],
    exact_ascii: dict[str, set[str]],
    ngrams: dict[str, set[str]],
    sparse_terms: dict[str, set[str]],
) -> None:
    if not alias.strip():
        return
    exact[normalize_text(alias)].add(concept_id)
    exact_ascii[normalize_text(alias, strip_accents=True)].add(concept_id)
    for gram in char_ngrams(alias):
        ngrams[gram].add(concept_id)
    for token in tokens(alias):
        sparse_terms[token].add(concept_id)


def _index_hash(index_payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(index_payload, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def build_icd_index(
    normalized_csv: str | Path,
    output_dir: str | Path,
    *,
    source_snapshot_id: str,
    source_hash: str = "",
) -> IndexMetadata:
    """Build an ICD index from the derived normalized CSV."""
    path = Path(normalized_csv)
    source = source_hash or _sha256_file(path)
    exact: dict[str, set[str]] = defaultdict(set)
    exact_ascii: dict[str, set[str]] = defaultdict(set)
    ngrams: dict[str, set[str]] = defaultdict(set)
    sparse_terms: dict[str, set[str]] = defaultdict(set)
    graph: dict[str, set[str]] = defaultdict(set)
    records: list[IndexRecord] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            concept_id = row["undotted_code"]
            aliases = tuple(
                a for a in (row.get("aliases") or "").split("|") if a
            )
            name = row.get("vietnamese_label", "")
            all_aliases = tuple(
                a for a in (name, row.get("dotted_code", ""), concept_id, *aliases) if a
            )
            for alias in all_aliases:
                _add_alias(
                    concept_id,
                    alias,
                    exact=exact,
                    exact_ascii=exact_ascii,
                    ngrams=ngrams,
                    sparse_terms=sparse_terms,
                )
            parent = row.get("parent", "")
            if parent:
                graph[parent].add(concept_id)
                graph[concept_id].add(parent)
            records.append(
                IndexRecord(
                    concept_id=concept_id,
                    canonical_name=name,
                    aliases=aliases,
                    metadata={
                        "dotted_code": row.get("dotted_code", ""),
                        "chapter": row.get("chapter", ""),
                        "block": row.get("block", ""),
                        "specificity": row.get("specificity", ""),
                    },
                )
            )
    records.sort(key=lambda r: r.concept_id)
    payload = {
        "records": [(r.concept_id, r.canonical_name, r.aliases, r.metadata) for r in records],
        "exact": {k: sorted(v) for k, v in _freeze_postings(exact).items()},
        "exact_ascii": {k: sorted(v) for k, v in _freeze_postings(exact_ascii).items()},
        "ngrams": {k: sorted(v) for k, v in _freeze_postings(ngrams).items()},
        "sparse_terms": {k: sorted(v) for k, v in _freeze_postings(sparse_terms).items()},
        "graph": {k: sorted(v) for k, v in _freeze_postings(graph).items()},
    }
    deterministic_hash = _index_hash(payload)
    metadata = IndexMetadata(
        index_id=f"icd10-index-{deterministic_hash[:16]}",
        index_type="icd10_vi",
        source_snapshot_id=source_snapshot_id,
        source_hash=source,
        config_hash=_config_hash({"builder": BUILDER_VERSION, "type": "icd10_vi"}),
        builder_version=BUILDER_VERSION,
        deterministic_index_hash=deterministic_hash,
        record_count=len(records),
        concept_count=len({r.concept_id for r in records}),
        parameters={"ngram_n": 3, "normalization": "casefold+NFKC+accent_optional"},
    )
    _write_index(
        SearchIndex(
            metadata=metadata,
            records=tuple(records),
            exact=_freeze_postings(exact),
            exact_ascii=_freeze_postings(exact_ascii),
            ngrams=_freeze_postings(ngrams),
            sparse_terms=_freeze_postings(sparse_terms),
            graph=_freeze_postings(graph),
        ),
        Path(output_dir),
    )
    return metadata


def _rrf_rows(path: Path) -> Iterator[list[str]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            cols = raw.rstrip("\n").split("|")
            if cols and cols[-1] == "":
                cols = cols[:-1]
            yield cols


def _get(cols: list[str], idx: int) -> str:
    return cols[idx] if idx < len(cols) else ""


def build_rxnorm_index(
    rrf_root: str | Path,
    output_dir: str | Path,
    *,
    source_snapshot_id: str,
    source_hash: str = "",
) -> IndexMetadata:
    """Build a local RxNorm index from RRF files without network access."""
    root = Path(rrf_root)
    source = source_hash or _tree_hash(root)
    exact: dict[str, set[str]] = defaultdict(set)
    exact_ascii: dict[str, set[str]] = defaultdict(set)
    ngrams: dict[str, set[str]] = defaultdict(set)
    sparse_terms: dict[str, set[str]] = defaultdict(set)
    graph: dict[str, set[str]] = defaultdict(set)
    aliases_by_cui: dict[str, set[str]] = defaultdict(set)
    meta_by_cui: dict[str, dict[str, str]] = defaultdict(dict)

    atom_count = 0
    for cols in _rrf_rows(root / "RXNCONSO.RRF"):
        atom_count += 1
        rxcui = _get(cols, _CONSO["RXCUI"])
        alias = _get(cols, _CONSO["STR"])
        if not rxcui or not alias:
            continue
        aliases_by_cui[rxcui].add(alias)
        meta = meta_by_cui[rxcui]
        meta.setdefault("tty", _get(cols, _CONSO["TTY"]))
        meta.setdefault("sab", _get(cols, _CONSO["SAB"]))
        meta.setdefault("suppress", _get(cols, _CONSO["SUPPRESS"]) or "N")
        if _get(cols, _CONSO["ISPREF"]) == "Y":
            meta["preferred_name"] = alias

    relation_count = 0
    for cols in _rrf_rows(root / "RXNREL.RRF"):
        relation_count += 1
        rxcui1 = _get(cols, _REL["RXCUI1"])
        rxcui2 = _get(cols, _REL["RXCUI2"])
        rela = _get(cols, _REL["RELA"])
        if rxcui1 and rxcui2 and rela in STRUCTURE_RELAS:
            graph[rxcui1].add(rxcui2)
            graph[rxcui2].add(rxcui1)

    attribute_count = 0
    for cols in _rrf_rows(root / "RXNSAT.RRF"):
        attribute_count += 1
        rxcui = _get(cols, _SAT["RXCUI"])
        atn = _get(cols, _SAT["ATN"])
        if rxcui and atn in _ATTRIBUTES_TO_KEEP:
            meta_by_cui[rxcui][atn] = _get(cols, _SAT["ATV"])

    records: list[IndexRecord] = []
    for rxcui, aliases in aliases_by_cui.items():
        for alias in aliases:
            _add_alias(
                rxcui,
                alias,
                exact=exact,
                exact_ascii=exact_ascii,
                ngrams=ngrams,
                sparse_terms=sparse_terms,
            )
        canonical = meta_by_cui[rxcui].get("preferred_name") or sorted(aliases)[0]
        records.append(
            IndexRecord(
                concept_id=rxcui,
                canonical_name=canonical,
                aliases=tuple(sorted(aliases)),
                metadata={k: v for k, v in sorted(meta_by_cui[rxcui].items())},
            )
        )
    records.sort(key=lambda r: r.concept_id)
    payload = {
        "records": [(r.concept_id, r.canonical_name, r.aliases, r.metadata) for r in records],
        "exact": {k: sorted(v) for k, v in _freeze_postings(exact).items()},
        "exact_ascii": {k: sorted(v) for k, v in _freeze_postings(exact_ascii).items()},
        "ngrams": {k: sorted(v) for k, v in _freeze_postings(ngrams).items()},
        "sparse_terms": {k: sorted(v) for k, v in _freeze_postings(sparse_terms).items()},
        "graph": {k: sorted(v) for k, v in _freeze_postings(graph).items()},
    }
    deterministic_hash = _index_hash(payload)
    metadata = IndexMetadata(
        index_id=f"rxnorm-index-{deterministic_hash[:16]}",
        index_type="rxnorm",
        source_snapshot_id=source_snapshot_id,
        source_hash=source,
        config_hash=_config_hash({"builder": BUILDER_VERSION, "type": "rxnorm"}),
        builder_version=BUILDER_VERSION,
        deterministic_index_hash=deterministic_hash,
        record_count=atom_count,
        concept_count=len(records),
        parameters={
            "ngram_n": 3,
            "relations_seen": relation_count,
            "attributes_seen": attribute_count,
            "normalization": "casefold+NFKC+accent_optional",
        },
    )
    _write_index(
        SearchIndex(
            metadata=metadata,
            records=tuple(records),
            exact=_freeze_postings(exact),
            exact_ascii=_freeze_postings(exact_ascii),
            ngrams=_freeze_postings(ngrams),
            sparse_terms=_freeze_postings(sparse_terms),
            graph=_freeze_postings(graph),
        ),
        Path(output_dir),
    )
    return metadata
