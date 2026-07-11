# L1 Offset Alignment

L1 keeps three representations (spec section 4):

1. **`original_text`** — the exact decoded input. **Immutable.** The **only**
   source of emitted `text`/`position`.
2. **normalized views** — transformed copies (e.g. `match`, `accent`) used for
   matching/retrieval. Every view stores a reversible alignment back to the
   original. **Normalized coordinates are never output coordinates.**
3. **token/segment graph** — structural nodes, each storing absolute offsets
   into `original_text`.

## `[start, end)` — end-exclusive, code points

Offsets are half-open intervals over Unicode **code points**, exactly Python
slicing `text[start:end]`. `end - start` equals the code-point length of the
span. This matches the organizer's confirmed convention
(`original_text[start:end] == text`).

## `CharAlignment`

`alignment.CharAlignment` stores a single monotonic **boundary map**:

- `o2n[i]` — normalized boundary for original boundary `i` (`0..original_length`),
  monotonic non-decreasing with `o2n[0] == 0` and `o2n[-1] == normalized_length`.

Covering original spans are derived by binary search over `o2n`
(`bisect`), so a normalized span always resolves to the **smallest valid covering**
original span. Because it is a boundary map (length N+1), the alignment supports:

- **one-to-one** (casefold of most letters),
- **one-to-many** (`ß` → `ss`; an original char → several normalized chars),
- **many-to-one** (`"a   b"` → `"a b"`; several original chars → one),
- **chars introduced only in a normalized view**, and
- **original chars omitted from a view** (deleted whitespace).

### APIs

```python
a.original_span_to_normalized(s, e)      # → normalized [ns, ne)
a.normalized_span_to_original(ns, ne)    # → smallest COVERING original [os, oe)
a.recover_original(original_text, ns, ne)# → exact original substring
a.then(other)                            # compose stage alignments
a.consistency_errors()                   # validate monotonicity/coverage
a.to_offset_alignment()                  # bridge to schemas.spans.OffsetAlignment
```

A normalized span always resolves to the **smallest valid covering** original
span (`os` = largest original boundary at/before the normalized start; `oe` =
smallest original boundary at/after the normalized end), so a match found on a
normalized view can always be re-expressed in authoritative original
coordinates. `recover_original` reads **only** from `original_text`, never from a
normalized view. These properties are checked exhaustively over every boundary
and span for representative transforms in
`tests/unit/test_l1_alignment_properties.py`.

## How alignment is built (O(n), exact)

Each normalization stage reports, per input character, how many output
characters it produced (`0` deleted, `1` kept/substituted, `>1` expanded).
`CharAlignment.from_counts` turns those counts into the `o2n` boundary map in
linear time — no text diffing, so long/repetitive inputs stay fast. Stage
alignments are composed with `then(...)` into a single original→view alignment.

`schemas.spans.OffsetAlignment` and `schemas.document.NormalizedView` are reused
(via `NormalizedDocument.as_schema_view()` / `to_offset_alignment()`); the richer
boundary API lives in `CharAlignment`.

## The one rule that matters

A later module may **read** a normalized view to match/retrieve, then map back to
a covering original span — but organizer output must always use the **original**
coordinates and the **original** substring. L1 never lets a normalized coordinate
become a final offset.
