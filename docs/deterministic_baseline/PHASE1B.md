# Phase 1B — Deterministic Medical Structure Specialists

Phase 1B is the first milestone that produces **medical proposals**. It is
deterministic, offset-preserving, and produces **no final organizer output**.

## Pipeline

```
input text ─L1─▶ DocumentGraph
   ─▶ Case Router (multi-label C1-C7)
   ─▶ medication grammar (C1/C5)   ─▶ SpanProposals + boundary candidates
   ─▶ laboratory parser (C2)       ─▶ TEST_NAME/TEST_RESULT + has_result relations
   ─▶ derive C6/C7 route evidence from proposals
   ─▶ deterministic merge (never by text) + diagnostics
   ─▶ validate every proposal against original_text
   ─▶ deterministic debug JSON
```

API: `run_phase1b(document_graph, Phase1BConfig) -> Phase1BResult`.

## Guarantees

- `original_text[start:end] == proposal.text` for every proposal and every
  structured component; normalized text never provides output offsets.
- Deterministic ids and ordering; the same input + configs yields byte-identical
  debug JSON (`determinism_hash`).
- **Proposals, not finals.** No final entity resolution, assertions, ICD/RxNorm
  codes, organizer JSON, or `output.zip`. `has_result` is internal only.
- Heuristic `local_score`s are **not** calibrated probabilities.

## Contracts

`SpanProposal` (id, document, `[start,end)`, text, proposed types, specialist,
source node, source routes, `local_score`, matched rule, normalized form,
components, `parse_ref`, evidence ids, config/lexicon versions, warnings,
features). `RelationProposal` (id, type, source/target proposal ids, score,
pairing cost, provenance). Merge keeps identical-span duplicates (preserving
provenance) and flags overlaps/repeats — it does **not** deduplicate by text.

## CLI

```bash
python -m mednorm_vi.deterministic_baseline.cli \
  --input tests/fixtures/phase1b/synthetic_medical_document.txt \
  --l1-config configs/document_intelligence/base.yaml \
  --router-config configs/case_router/base.yaml \
  --medication-config configs/medication/grammar_v1.yaml \
  --laboratory-config configs/laboratory/parser_v1.yaml \
  --output /tmp/mednorm-phase1b.json --inspect
```

## Consumed by the Phase 1C-A resolver

The deterministic L4 Boundary & Type Resolver foundation (Phase 1C-A) consumes
these proposals + relations and produces `EntityHypothesis` objects (chosen
boundary, retained alternatives, accepted/rejected/unresolved). It performs no
ontology linking and emits no final entities. See `docs/resolution/RESOLVER.md`.

## What remains (Phase 1C / Phase 2)

Assertion classification, ICD-10 / RxNorm linking over frozen permitted KB
snapshots (foundation in `docs/kb/`), the evidence graph, and the first
organizer-compatible baseline output. See `docs/architecture/LAYER_CONTRACTS.md`.
