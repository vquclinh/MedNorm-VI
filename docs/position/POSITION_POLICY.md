# Submission Position Policy (Phase 1C-A)

## Two coordinate systems, on purpose

MedNorm-VI keeps **two** position representations:

1. **Internal raw code-point spans** — the single source of truth. The invariant
   `original_text[start:end] == text` is enforced by L1 and by every proposal /
   resolver validator, and Phase 1C-A does **not** weaken it. All internal
   reasoning, diagnostics, and the default output use these coordinates
   (`raw-codepoint-half-open`).

2. **Submission-encoded positions** — a *separate, reversible* re-expression of a
   raw span produced by a registered `PositionPolicy`. This is an adapter, not a
   mutation: an encoder changes only coordinates, never entity text, and every
   `PositionEncodingResult` carries `PositionProvenance` back to the raw span.

## Why an adapter instead of a single coordinate space

The organizer's graded coordinate space is **unresolved** (see
`configs/organizer/unresolved_policies_v1.yaml`: `UNRES-POS-COORD-SPACE`,
`UNRES-POS-CHAR-VS-BYTE`, `UNRES-POS-LINE-ENDING`, `UNRES-POS-INTERVAL`,
`UNRES-MATCH-MODE`). Committing the whole pipeline to one guess would be fragile.
Instead we keep raw offsets strict internally and treat submission coordinates as
a swappable policy we can probe on the leaderboard.

## Registered policies

`configs/organizer/position_policies_v1.yaml` (default `raw-codepoint-half-open`):

- `raw-codepoint-half-open` — identity; internal + diagnostic default.
- `utf8-byte-half-open` — UTF-8 byte offsets.
- `canonical-lf-codepoint` — offsets after CRLF→LF.
- `canonical-crlf-codepoint` — offsets after lone LF→CRLF.
- `separator-reconstruction-codepoint` — configurable line separator.
- `normalized-view-codepoint` — offsets into the L1 normalized view; available
  ONLY when a reversible `OffsetAlignment` exists, else it fails clearly.

Each encoder builds a monotonic forward map; decoding reverses it and **fails
clearly** if a target (e.g. a byte offset inside a multi-byte character) does not
land on a code-point boundary — it never silently rounds.

## Forensics

`position-forensics` compares every registered policy against a *user-provided*
sample input/output pair and reports exact/near matches, systematic deltas,
per-line cumulative shift, max offset, and byte-vs-code-point / CRLF-vs-LF /
inclusive-vs-exclusive evidence. It contains **no** organizer sample text; supply
paths:

    python -m mednorm_vi.phase1c_foundation.cli position-forensics \
        --input sample_input.txt --output sample_output.json

## Validation guarantees

`validate_encoding` checks: the policy is registered; the raw invariant holds;
re-encoding is deterministic; and (where the policy claims reversibility) the
encode→decode round-trip recovers the exact raw span.
