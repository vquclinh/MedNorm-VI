#!/usr/bin/env python3
"""Project existing raw predictions onto the organizer schema (sprint 0075 hotfix).

One-shot, deterministic, in-place-safe migration so an expensive inference run can be reused
instead of re-executed. It ONLY removes fields the organizer schema does not allow for a
given entity type. No value is altered, no entity is added or removed, and entity order is
preserved exactly.

    CHẨN_ĐOÁN / THUỐC        text, type, position, assertions, candidates
    TRIỆU_CHỨNG              text, type, position, assertions
    TÊN_XÉT_NGHIỆM           text, type, position
    KẾT_QUẢ_XÉT_NGHIỆM       text, type, position
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.reasoner.validator import (  # noqa: E402
    ASSERTION_TYPES,
    CANDIDATE_TYPES,
    ORGANIZER_TYPES,
)

BASE_FIELDS = ("text", "type", "position")


def project(entity: dict[str, Any]) -> dict[str, Any]:
    """The allowed fields for this entity's type, in organizer order. Values untouched."""
    entity_type = entity.get("type", "")
    row: dict[str, Any] = {field: entity[field] for field in BASE_FIELDS if field in entity}
    if entity_type in ASSERTION_TYPES and "assertions" in entity:
        row["assertions"] = entity["assertions"]
    if entity_type in CANDIDATE_TYPES and "candidates" in entity:
        row["candidates"] = entity["candidates"]
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-documents", type=int, default=100)
    args = parser.parse_args(argv)

    files = sorted(args.source_dir.glob("*.json"))
    if len(files) != args.expected_documents:
        print(f"BLOCKED: found {len(files)} documents, expected {args.expected_documents}",
              file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    removed: Counter[str] = Counter()
    kept: Counter[str] = Counter()
    entities = 0
    for path in files:
        rows = json.loads(path.read_text(encoding="utf-8"))
        out = []
        for entity in rows:
            entities += 1
            entity_type = entity.get("type", "")
            if entity_type not in ORGANIZER_TYPES:
                print(f"BLOCKED: {path.name} has unknown type {entity_type!r}", file=sys.stderr)
                return 3
            projected = project(entity)
            for field in entity:
                if field not in projected:
                    removed[f"{entity_type}.{field}"] += 1
            kept[f"{entity_type}:{','.join(sorted(projected))}"] += 1
            out.append(projected)
        if len(out) != len(rows):
            print(f"BLOCKED: {path.name} entity count changed", file=sys.stderr)
            return 4
        (args.output_dir / path.name).write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8"
        )

    print(f"documents {len(files)} | entities {entities} (unchanged)")
    print("removed unsupported fields:")
    for key, count in sorted(removed.items()):
        print(f"  {key:38s} {count:5d}")
    if not removed:
        print("  (none - input already conformed)")
    print("resulting shapes:")
    for key, count in sorted(kept.items()):
        print(f"  {key:56s} {count:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
