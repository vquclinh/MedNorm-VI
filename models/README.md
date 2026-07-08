# models/

Root for **self-hosted base-model weights and adapters**. Everything under this
directory except this README is git-ignored — **never commit weights**.

- All models are local/offline; the pipeline makes **no external API calls**.
- Base-model parameters count toward the **9B budget**; adapters/LoRA/heads do
  not. Every model (base and adapter) must appear in
  `configs/parameter_budget.yaml`, verified by `mednorm_vi.validator.budget`.
- Keep base weights and adapters clearly separated, e.g.:

```
models/
├── base/       # base checkpoints (count toward 9B)
└── adapters/   # LoRA / task heads (excluded from 9B, still manifested)
```

No weights are downloaded at the bootstrap stage.
