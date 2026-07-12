# Laboratory Lexicon Policy (Phase 1B)

## No downloads

No laboratory ontology is downloaded. A small representative **seed** lexicon
(`configs/laboratory/test_lexicon_seed_v1.yaml`, `provenance: representative_seed`)
with canonical names + aliases, plus a versioned unit inventory
(`units_v1.yaml`), ships for tests and bootstrapping.

## Extensible, local production lexicon

The seed declares `production_lexicon_path` (default `data/lexicons/lab_tests.txt`)
— a **local, git-ignored** file to be populated later from permitted sources. If
present it is merged at load time. It is absent here. Do not embed a final
exhaustive lab lexicon in Python.

## Unknown tests

An unknown test name is proposed only in a strongly structured row and carries an
`unknown_test_name` warning; narrative parsing requires a known lexicon test.

## Versioning & provenance

`lexicon_version` / `parser_version` are recorded on every proposal. Units are
structural evidence; aliases support Vietnamese, English, and abbreviations
(e.g. `HGB`↔`Hb`, `Glucose`↔`đường huyết`), all versioned and extensible.
