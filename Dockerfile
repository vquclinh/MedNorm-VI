# MedNorm-VI — offline inference image (spec §19, Appendix A) — Audit 0053.
#
# Appendix A's requirements this image exists to satisfy:
#   * "Every model/tokenizer/KB/index uses a local path; inference requires no
#     external API."
#   * "One command runs from the input directory to output.zip."
#
# NOT BUILT by Audit 0053. This file is authored as source/environment
# reproducibility infrastructure; building it, and any organizer inference, is a
# separate authorized step. Nothing here downloads a model.
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
#   /models/e3/best.pt            the E3 checkpoint
#                                 sha256 a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
#   /models/hf                    HF cache holding the ViHealthBERT config +
#                                 tokenizer at revision f89e80b4…24b6c5
#                                 (weights are NOT needed: the checkpoint carries
#                                 the complete state dict)
#   /models/vncorenlp             VnCoreNLP-1.2.jar + models
#   /kb/indices                   generated ICD-10 / RxNorm index.json files
#   /input                        organizer .txt documents          (read-only)
#   /output                       destination for output/ and output.zip
#
# Example (illustrative; do not run as part of this milestone):
#   docker build -t mednorm-vi:0053 .
#   docker run --rm --network=none \
#     -v "$PWD/checkpoint/s1_mention_full_training_v1:/models/e3:ro" \
#     -v "$HOME/.cache/huggingface:/models/hf:ro" \
#     -v "$HOME/.cache/mednorm-vi/vncorenlp:/models/vncorenlp:ro" \
#     -v "$PWD/indices:/kb/indices:ro" \
#     -v "$PWD/data/organizer_test/input:/input:ro" \
#     -v "$PWD/outputs:/output" \
#     mednorm-vi:0053 run --input-dir /input --output-zip /output/output.zip \
#       --config configs/pipeline/full_v1.yaml --mode deterministic
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
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/app/src \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/models/hf \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MEDNORM_E3_CHECKPOINT=/models/e3/best.pt \
    MEDNORM_VNCORENLP_DIR=/models/vncorenlp

WORKDIR /app

# Dependencies first, from the lock file only, so a source change does not
# re-resolve the environment. `--no-deps` is safe because requirements.lock
# enumerates the full graph (see its header).
COPY requirements.lock ./
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir --no-deps -r requirements.lock

# Source, configs and the schema/contract docs. No data, no weights, no notebooks.
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY schemas/ ./schemas/

# Non-root runtime. The mount points are created and owned before the drop so a
# read-only bind works and /output stays writable.
RUN mkdir -p /models /kb /input /output \
 && useradd --create-home --shell /usr/sbin/nologin --uid 10001 mednorm \
 && chown -R mednorm:mednorm /app /output
USER mednorm

# Fail fast on a broken image: the package must import and the L1-L9 contracts
# must resolve with no model and no network.
RUN python -c "import mednorm_vi, mednorm_vi.inference.pipeline, mednorm_vi.validator.kb_membership; print('mednorm-vi import ok')"

# ONE deterministic entry point: the canonical inference CLI. There is no
# per-model entry point, because there is one runner (Audit 0052) and experts
# arrive through the L3 registry rather than through separate commands.
ENTRYPOINT ["python", "-m", "mednorm_vi.inference.cli"]
CMD ["--help"]
