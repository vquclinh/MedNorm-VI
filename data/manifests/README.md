# data/manifests/ — tracked resource manifests (payloads stay untracked)

Filled-in resource manifests (one per acquired resource) live here and ARE
tracked — they contain only governance metadata (version, source, checksums,
license review, intended use), never resource content.

Start from a template in `configs/resources/`, fill it in after manual
acquisition, and validate:

    python -m mednorm_vi.phase1c_foundation.cli validate-resource-manifest \
        --manifest data/manifests/<resource>.yaml
