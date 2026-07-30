# MedNorm-VI — offline inference image (spec §19, Appendix A) — Audit 0053.
#
# Appendix A's requirements this image exists to satisfy:
#   * "Every model/tokenizer/KB/index uses a local path; inference requires no
#     external API."
#   * "One command runs from the input directory to output.zip."
#
# BUILT AND OFFLINE-SMOKE-TESTED by Audit 0054. Audit 0053 authored this file without
# building it; the first real build failed at the pip step, because
# `requirements.lock` pins `torch==2.13.0+cu126` — a local version identifier that
# does not exist on PyPI. The image now installs `requirements-image.lock`, the same
# upstream torch version from the CPU index, and that file records why.
#
# Nothing here downloads a model. No organizer inference runs. No output.zip is made.
#
# --- What is deliberately NOT in the image ---------------------------------
#   * model weights — never in Git and never baked in (see MOUNTS below);
#   * the governed corpus and KB payloads — licence-restricted (RxNorm is
#     UMLS-licensed and redistribution-restricted per data/manifests/);
#   * any network step at runtime. `local_files_only=True` is enforced in
#     `llm/backends.py` and `mention_factory/neural/runtime.py`, and
#     `HF_HUB_OFFLINE=1` below makes an accidental fetch fail loudly instead of
#     silently succeeding.
#
# --- MOUNTS an authorized build/run must supply ---------------------------
#
# CORRECTED BY AUDIT 0055 after actually running the image. Two things in the
# Audit-0053 contract were wrong, and only a real run could show it:
#
#   1. the mount points must match the paths the ACTIVE PROFILE resolves.
#      `configs/pipeline/full_v1.yaml` declares `indices/...` and E3 defaults to
#      `checkpoint/...`, both relative — and WORKDIR is /app. So they mount at
#      /app/indices and /app/checkpoint, NOT at the /kb/indices and /models/e3
#      paths Audit 0053 documented. With the old paths, `deterministic` mode
#      reported NOT_READY (missing_icd_index, missing_rxnorm_index) — fail-closed,
#      correctly, but for a reason that looked like a missing asset rather than a
#      wrong contract. A profile that points at /kb and /models is equally valid;
#      the paths simply have to agree with each other.
#
#   2. on an SELinux-enforcing host, every bind mount needs a relabel suffix.
#      Without it /output is unwritable at mode 0777 and /input is unreadable,
#      because the denial is a label mismatch and has nothing to do with the mode
#      bits. Use `:ro,Z` for assets and `:Z` for the output directory.
#
#   /app/checkpoint     E3 checkpoint tree (read-only)
#                       s1_mention_full_training_v1/best.pt
#                       sha256 a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
#   /app/indices        generated ICD-10 / RxNorm index.json files (read-only)
#   /models/hf          HF cache holding the ViHealthBERT config + tokenizer at
#                       revision f89e80b4…24b6c5 (read-only). The WEIGHTS are not
#                       needed: the checkpoint carries the complete state dict.
#   /models/vncorenlp   VnCoreNLP-1.2.jar + models (read-only)
#   /input              organizer .txt documents (read-only)
#   /output             destination for output/ and output.zip — the ONLY writable
#                       mount
#
# The command Audit 0055 actually ran, offline, for the specialist/E3 smoke. Drop
# the checkpoint/hf/vncorenlp mounts for a `deterministic` run — E3 is not used.
#
#   docker build --pull -t mednorm-vi:framework-closed .
#   docker run --rm --network=none \
#     -v "$PWD/checkpoint:/app/checkpoint:ro,Z" \
#     -v "$PWD/indices:/app/indices:ro,Z" \
#     -v "$HOME/.cache/huggingface:/models/hf:ro,Z" \
#     -v "$HOME/.cache/mednorm-vi/vncorenlp:/models/vncorenlp:ro,Z" \
#     -v "$PWD/data/organizer_test/input:/input:ro,Z" \
#     -v "$PWD/outputs:/output:Z" \
#     mednorm-vi:framework-closed run --input-dir /input \
#       --output-zip /output/output.zip \
#       --config configs/pipeline/full_v1.yaml --mode specialist \
#       --run-manifest /output/run-manifest.json
#
# Drop `,Z` on a host without SELinux. Keep `:ro` regardless: Audit 0055 verified
# every asset mount returns EROFS from inside the container, including the
# checkpoint directory.
#
# `--network=none` is the point: if any path tried to reach the network, the run
# would fail rather than quietly download.

FROM python:3.14.5-slim-bookworm

# A JRE is required by py_vncorenlp (E3's segmentation contract). No other system
# package is installed, so the attack surface stays close to the base image.
RUN apt-get update \
 && apt-get install --no-install-recommends -y openjdk-17-jre-headless \
 && rm -rf /var/lib/apt/lists/*

# Offline and deterministic by construction.
# HF_HOME points at the mounted cache; the E3 adapter and the KB indices resolve
# through the ACTIVE PROFILE's relative paths under WORKDIR (see MOUNTS above).
#
# MEDNORM_E3_CHECKPOINT and MEDNORM_VNCORENLP_DIR were declared here by Audit 0053
# and REMOVED by Audit 0055: nothing in the source ever read them, and they pointed
# at /models/e3 and /models/vncorenlp, which are not where the active profile looks.
# A configuration variable no code reads is worse than no variable — it tells an
# operator they have configured something when they have not.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/app/src \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/models/hf \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1

WORKDIR /app

# Torch variant, selectable and explicit. The default is the CPU wheel, because every
# recorded E3 validation run in Audits 0052-0054 was a CPU forward pass and the CUDA
# runtime would add ~2.5 GB the inference path never touches. A GPU image is the same
# Dockerfile with both args overridden:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
#   --build-arg TORCH_SPEC=torch==2.13.0+cu126
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_SPEC=torch==2.13.0

# Dependencies first, from the lock file only, so a source change does not re-resolve
# the environment. `--no-deps` is safe because requirements-image.lock enumerates the
# full graph (see its header). Both indexes are named explicitly rather than inherited
# from ambient pip configuration, so the build cannot silently resolve a different
# torch build than the one recorded.
COPY requirements-image.lock ./
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir --no-deps \
      --index-url "${TORCH_INDEX_URL}" \
      --extra-index-url https://pypi.org/simple \
      "${TORCH_SPEC}" \
 && python -m pip install --no-cache-dir --no-deps \
      --index-url "${TORCH_INDEX_URL}" \
      --extra-index-url https://pypi.org/simple \
      -r requirements-image.lock \
 && python -c "import torch; print('torch', torch.__version__)"

# Source, configs and the schema/contract docs. No data, no weights, no notebooks.
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY schemas/ ./schemas/

# Non-root runtime. The mount points are created and owned before the drop so a
# read-only bind works and /output stays writable.
# `/models`, `/kb` and `/input` are mounted READ-ONLY in the documented run command;
# only `/output` is writable, and only by the non-root user.
RUN mkdir -p /models /kb /input /output \
 && useradd --create-home --shell /usr/sbin/nologin --uid 10001 mednorm \
 && chown -R mednorm:mednorm /app /output \
 && chmod 0755 /models /kb /input
USER mednorm

# Fail fast on a broken image: the package must import and the L1-L9 contracts
# must resolve with no model and no network.
RUN python -c "import mednorm_vi, mednorm_vi.inference.pipeline, mednorm_vi.validator.kb_membership; print('mednorm-vi import ok')"

# ONE deterministic entry point: the canonical inference CLI. There is no
# per-model entry point, because there is one runner (Audit 0052) and experts
# arrive through the L3 registry rather than through separate commands.
ENTRYPOINT ["python", "-m", "mednorm_vi.inference.cli"]
CMD ["--help"]
