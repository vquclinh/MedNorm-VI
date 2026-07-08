# ADR 0001 — Open questions pending organizer data

- **Status:** Open (partially resolved — see "Resolved" below)
- **Date:** 2026-07-08 (updated 2026-07-08, Audit 0002)
- **Context:** The architecture spec (section 21.1) lists decisions that cannot
  be resolved until the organizer's data and annotation convention are known.
  These are recorded here so no module hard-codes a guess.

## Resolved (confirmed by the organizer — no longer open)

- **R1. Offset convention.** Python-style, **end-exclusive** character offsets
  over Unicode code points: `original_text[start:end] == entity["text"]`.
  Golden example: text `amlodipine 10 mg po daily`, position `[58, 83]`,
  length 25. Enforced by `validator/offsets.py`; golden test in `test_offsets.py`.
- **R2. Submission file count, names, and directory layout.**
  `output.zip` → one top-level `output/` directory → `1.json` … `100.json`
  (exactly 100, no missing IDs, no extra prediction JSON). Enforced by
  `validator/submission.py` (`validate_output_directory`, `validate_submission_zip`).
- **R3. Organizer entity labels & per-type fields.** Vietnamese `type` labels
  and per-type output fields are fixed (see `README.md` / `constants.py`).
  Enforced by `validator/organizer.py`.

## Still undecided (do not guess — verify against organizer data)

1. Does entity matching use position, or only `text` + `type`?
2. What exact tokenization/normalization is used for Word Error Rate?
3. Which ICD-10 version and RxNorm release are provided (for frozen KB hashes)?
4. Is the candidate list length limited, and does candidate ordering matter?
5. Are nested/overlapping entities present in the gold data?
6. Will later rounds include an explicit relation field?
7. Are there time/memory limits for the private-test rebuild?
8. Are suppressed/historical ontology concepts accepted, or active-only?

## Consequences for the bootstrap

- `configs/base.yaml` leaves KB versions/hashes as `null` (to be filled).
- The duplicate policy in `validator/duplicates.py` keys on `(text, type,
  start, end)` — i.e. position-aware — because the spec forbids text-only
  deduplication (section 7.3, C7). Revisit if organizer matching is text+type only.
- Candidate **syntax** is kept light (RxNorm = numeric RxCUI strings; ICD-10 =
  opaque strings). **KB-membership** validation remains deferred until frozen KB
  releases exist (question 3 above).

## Decision-record convention

Add a new numbered ADR (`NNNN-title.md`) for any architecture-sensitive choice.
Reference the relevant spec section. Never delete an ADR; supersede it with a
new one and mark the old one `Superseded by ADR NNNN`.
