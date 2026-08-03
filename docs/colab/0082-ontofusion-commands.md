# 0082 — OntoFusion linking (pretrained-only)

A **second linker** beside GraphCENT 0080, never a replacement, so each run changes one
variable:

| run | entities | linker |
|---|---|---|
| 0080 | E3 | GraphCENT |
| 0081 | contextual (0081) | GraphCENT |
| 0082 | contextual (0081) | **OntoFusion** |

Nothing is trained, fine-tuned or adapted, and no model is added. The deployed stack is
exactly the certified GraphCENT stack — **8,729,759,237 parameters** — and sparse character
retrieval contributes none. `ontofusion_0082.py budget` re-proves this from the canonical
`model-manifest.json` and refuses to continue if the total has moved.

**The existing 0080 semantic caches stay valid.** 0082 changes no encoder, no pooling, no KB
document text and no row order; it reads the caches through GraphCENT's own loader and never
writes one. Run cells 1–4 of [0080-graphcent-commands.md](0080-graphcent-commands.md) and
then [0081-contextual-commands.md](0081-contextual-commands.md) first.

## Qwen lifecycle and memory

```
load Qwen  -> reformulate every unique mention -> UNLOAD
load each retriever in turn -> retrieve -> UNLOAD each
load Qwen  -> set-wise rerank every mention    -> UNLOAD
```

Qwen is loaded twice rather than held resident across retrieval, so peak VRAM is one model at
a time (~17 GiB) and an L4 is a supported host. Peak allocation is measured and reported in
the diagnostics.

## Cell A — budget check

```bash
%%bash
export PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/ontofusion_0082.py budget \
  --model-root /content/graphcent_models \
  --expect-total 8729759237
```

## Cell B — 3-document 0081 smoke (produces the entities 0082 consumes)

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/contextual_0081.py smoke --allow-download \
  --input-dir data/organizer_test/input \
  --seed-entities runs/e3_verified_8b_allnull_0076/output \
  --qwen-root /content/qwen3-8b-local \
  --run-dir runs/contextual_0081/_smoke --documents 1,10,100
python -c "
import json;d=json.load(open('runs/contextual_0081/_smoke/diagnostic-summary.json'))
print(json.dumps(d,indent=2,ensure_ascii=False))"
```

## Cell C — 3-document 0082 smoke using `contextual_high`

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/ontofusion_0082.py smoke --allow-download \
  --input-dir data/organizer_test/input \
  --entities runs/contextual_0081/_smoke/entities_contextual_high \
  --model-root /content/graphcent_models --cache-root /content/graphcent_cache \
  --qwen-root /content/qwen3-8b-local \
  --run-dir runs/ontofusion_0082/_smoke --documents 1,10,100
python -c "
import json;d=json.load(open('runs/ontofusion_0082/_smoke/diagnostic-summary.json'))
print(json.dumps(d,indent=2,ensure_ascii=False))"
```

Read before going further: `unique_reformulations` and `reformulation_parse_failures`,
`canonical_vi_non_empty` / `canonical_en_non_empty`, `union_size_mean`, the
`retrieved_<retriever>` and `retrieved_view_<view>` counts, the prune reasons
(`strength_conflict`, `dose_form_conflict`, `wrong_ontology`, `not_governed`),
`select` versus `none`, `reranker_returned_a_code` and `reranker_invalid_option` (both should
be near zero), `high_selections`, `multi_code_selections` (expected **0**), and
`peak_vram_gib`.

Inspect individual decisions — the per-mention record holds the context, both reformulations,
every candidate with which retrievers found it, the option order and the selection, and no
chain-of-thought:

```bash
%%bash
cd /content/MedNorm-VI
python -c "
import json
for line in open('runs/ontofusion_0082/_smoke/mention-records.jsonl'):
    r=json.loads(line)
    print(r['type'], repr(r['mention']), '->', r['selection']['decision'],
          r['selection']['concept_id'], 'high' if r['high_confidence'] else '')
    print('    vi:', r['reformulation'].get('canonical_vi'),
          '| en:', r['reformulation'].get('canonical_en'))
    print('    options:', r['option_order'])
"
```

## Cell D — full run and packaging

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/contextual_0081.py run --allow-download \
  --input-dir data/organizer_test/input \
  --seed-entities runs/e3_verified_8b_allnull_0076/output \
  --qwen-root /content/qwen3-8b-local --run-dir runs/contextual_0081

python scripts/ontofusion_0082.py run --allow-download \
  --input-dir data/organizer_test/input \
  --entities runs/contextual_0081/entities_contextual_high \
  --model-root /content/graphcent_models --cache-root /content/graphcent_cache \
  --qwen-root /content/qwen3-8b-local --run-dir runs/ontofusion_0082

python scripts/ontofusion_0082.py package --run-dir runs/ontofusion_0082 \
  --input-dir data/organizer_test/input --expected-documents 100
cat runs/ontofusion_0082/package-summary.json
```

Packaging reuses GraphCENT's dotted derivation, organizer validation, zipping and hashing, so
a 0082 submission passes exactly the same gate as an 0080 one.

## Submission order

`output_allnull.zip` is the control and should reproduce the known all-null score — submit it
first to confirm the entity set did not change anything by itself. Then `ontofusion_high`
(selections with independent deterministic support), and only then `ontofusion_broad`.

The old Tier-A threshold is **not** reused: it was measured against the noisy E3 entity set
and is stale now that 0081 changed entity precision. Every input to the `high` policy is
stored per mention, so another threshold can be derived from `mention-records.jsonl` without
re-running Qwen.
