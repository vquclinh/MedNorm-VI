# 0081 — contextual proposal + verification, then GraphCENT 0080

0081 fixes the **extraction** bottleneck. GraphCENT 0080 is unchanged and still does all
candidate linking; 0081 only decides which mentions it is asked to link.

The deployed model is the **same Qwen3-8B** GraphCENT already loads, at the revision the
canonical `model-manifest.json` pinned. No second base model enters the deployment, so the
`< 9B` accounting is the 0080 accounting — `contextual_0081.py budget` re-proves it from that
manifest rather than recomputing anything.

```
document
  → E3 proposals + Qwen document-context proposals + governed exact-alias proposals
  → deterministic exact-source alignment      (offsets computed locally, never by the model)
  → finite overlap/boundary lattice
  → Qwen finite verifier                      (accept / reject / retype, by index only)
  → deterministic span+type resolver
  → contextual assertion pass
  → GraphCENT 0080 candidate linking          (unchanged)
```

Run cells 1–4 of [0080-graphcent-commands.md](0080-graphcent-commands.md) first: 0081 needs
the same repo, the same Qwen snapshot, the same governed indices, and GraphCENT still needs
its semantic caches. **The 0080 caches do not need rebuilding** — 0081 changes no encoder, no
pooling, no KB document text and no row order.

## Cell A — prove the budget is unchanged

```bash
%%bash
export PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/contextual_0081.py budget --model-root /content/graphcent_models
```

## Cell B — 3-document smoke

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

Read before going further: `final_entities_by_proposal_source` (how much is E3-only,
Qwen-only, alias-only, agreement), `verifier_accepted` / `verifier_rejected`,
`text_not_in_source` (Qwen proposals that did not align — these were rejected, not repaired),
`result_without_test`, `overlap_resolved`, and the `isNegated` / `isFamily` / `isHistorical`
counts.

Inspect a document directly:

```bash
%%bash
cd /content/MedNorm-VI
python -c "
import json
for line in open('runs/contextual_0081/_smoke/proposal-records.jsonl'):
    r=json.loads(line)
    print(r['document'], len(r['proposals']), '->', len(r['entities']))
    for e in r['entities'][:12]:
        print('   ', e['type'], repr(e['text']), sorted(e['sources']), e['assertions'])
"
```

## Cell C — link one variant with GraphCENT (still a smoke)

`entities_<variant>` is exactly the shape `--seed-entities` consumes.

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/graphcent_0080.py smoke --allow-download \
  --input-dir data/organizer_test/input \
  --seed-entities runs/contextual_0081/_smoke/entities_contextual_high \
  --model-root /content/graphcent_models --cache-root /content/graphcent_cache \
  --run-dir runs/graphcent_0080/_smoke_0081 --documents 1,10,100
```

## Cell D — full run, then link each variant

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/contextual_0081.py run --allow-download \
  --input-dir data/organizer_test/input \
  --seed-entities runs/e3_verified_8b_allnull_0076/output \
  --qwen-root /content/qwen3-8b-local --run-dir runs/contextual_0081

for V in e3_control contextual_high contextual_broad; do
  python scripts/graphcent_0080.py run --allow-download \
    --input-dir data/organizer_test/input \
    --seed-entities runs/contextual_0081/entities_$V \
    --model-root /content/graphcent_models --cache-root /content/graphcent_cache \
    --run-dir runs/graphcent_0080_$V
  python scripts/graphcent_0080.py package --run-dir runs/graphcent_0080_$V \
    --input-dir data/organizer_test/input --expected-documents 100
done
```

## Ablation order

`e3_control` reproduces the current entity set exactly, so linking it must reproduce the
known score — submit it first as the control. Only if it does should `contextual_high`
(contextual entities with independent agreement) and then `contextual_broad` be read as
evidence about extraction rather than about a pipeline change.

Within each entity variant, GraphCENT still emits its own four candidate tiers, so submit
the `tierA` zip first as before — all-null remains the public best at 14.3749.
