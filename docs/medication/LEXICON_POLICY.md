# Medication Lexicon Policy (Phase 1B)

## No downloads

Phase 1B downloads **no** RxNorm or external drug database. Only a small,
deterministic, representative **seed** ingredient list ships in
`configs/medication/lexicon_seed_v1.yaml` (`provenance: representative_seed`),
sufficient for tests and bootstrapping.

## Extensible, local production lexicon

The seed declares `production_lexicon_path` (default
`data/lexicons/medication_names.txt`) — a **local, git-ignored** file to be
populated later from permitted sources. If present, its entries (one lowercase
name per line, `#` comments allowed) are merged at load time. It is absent here.

Do **not** hard-code a supposedly exhaustive medication inventory in Python.

## Unknown names

A token not in the lexicon is proposed as a medication name **only** with
sufficient context evidence (a following strength / dose form / route), and the
proposal carries an `unknown_medication_name` warning. This keeps recall high
without inventing drugs from arbitrary words. Router gating (C1/C5) further
limits where the grammar runs.

## Versioning & provenance

`lexicon_version` and `grammar_version` are recorded on every proposal
(`config_version`, `lexicon_version`). Abbreviation expansions
(`abbreviations_v1.yaml`) are stored as **hints/evidence**, never a committed
final meaning.
