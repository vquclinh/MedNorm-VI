"""E3 ViHealthBERT span/type expert, as a registered L3 expert (Audit 0052).

E3 has been the only trained model in this project since Audit 0031, and until now
it could not run in production: the canonical runner had no path for a neural
expert. This adapter is that path. It changes nothing about the model — it wraps the
existing, unmodified :mod:`mednorm_vi.mention_factory.neural.runtime` and turns its
``NeuralSpan`` output into the common :class:`ExpertSpanProposal` contract.

What it deliberately does **not** do:

* **it never downloads the backbone.** ``load_expert`` rebuilds the architecture from
  the locally cached ``config.json`` at the pinned revision and loads every parameter
  from the checkpoint's own complete state dict, with ``local_files_only=True``;
* **it never writes to the checkpoint.** The file is opened read-only, its digest is
  verified before use, and re-verified after inference;
* **it never emits a final entity.** Proposals go into the lattice like every other
  expert's, and L4 decides.

Loading is lazy: constructing this object touches no model, so a profile with
``enable_e3_vihealthbert: false`` costs nothing.

VnCoreNLP is optional in the shipped runtime profile. When
``e3_vncorenlp_dir`` is absent, E3 uses the existing pre-segmented-source path.
When the setting is explicit, the configured VnCoreNLP capability is mandatory
for that run and readiness fails closed if it cannot be imported or constructed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...case_router.models import NodeRouting
from ...document_intelligence.models import DocumentGraph
from ...lattice.models import EXPERT_VIHEALTHBERT, ExpertSpanProposal
from ..registry import ExpertRegistration, L3Expert, register

E3_EXPERT_CONTRACT_VERSION = "e3-vihealthbert-expert-v1"
E3_ROLE = "l3_e3_mention_span_type"
E3_FEATURE_FLAG = "enable_e3_vihealthbert"
E3_CHECKPOINT_KEY = "mention/vihealthbert"

# The validated artifact (Audits 0031, 0032, 0051 §7). Both are asserted before the
# checkpoint is read, so a substituted or truncated file fails closed rather than
# producing plausible-looking proposals from the wrong weights.
E3_CHECKPOINT_SHA256 = "a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c"
E3_PINNED_MODEL_REVISION = "f89e80b461e86f9cfc1c84019bd819830c24b6c5"
E3_HF_MODEL_ID = "demdecuong/vihealthbert-base-word"

DEFAULT_CHECKPOINT_PATH = "checkpoint/s1_mention_full_training_v1/best.pt"


class E3VietHealthBertExpert:
    """The trained E3 span/type expert behind the L3 registry contract."""

    expert_id = EXPERT_VIHEALTHBERT
    role = E3_ROLE

    def __init__(
        self,
        *,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        expected_checkpoint_sha256: str = E3_CHECKPOINT_SHA256,
        pinned_model_revision: str = E3_PINNED_MODEL_REVISION,
        hf_model_id: str = E3_HF_MODEL_ID,
        model_cache_dir: str = "",
        vncorenlp_dir: str = "",
        decision_threshold: float = 0.5,
        max_sequence_length: int = 256,
        batch_size: int = 8,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self.pinned_model_revision = pinned_model_revision
        self.hf_model_id = hf_model_id
        self.model_cache_dir = model_cache_dir
        self.vncorenlp_dir = vncorenlp_dir
        self.decision_threshold = decision_threshold
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self._expert: Any = None
        self.model_revision = pinned_model_revision
        self.checkpoint_sha256 = expected_checkpoint_sha256

    # -- readiness -------------------------------------------------------------
    def readiness(self) -> tuple[bool, str, str]:
        """Whether E3 can run. Loads no model and reads no weights."""
        path = Path(self.checkpoint_path)
        if not self.checkpoint_path:
            return (False, "no E3 checkpoint path is configured", "")
        if not path.is_file():
            return (False, "the E3 checkpoint file does not exist", str(path))
        if not self.pinned_model_revision:
            return (False, "no pinned backbone revision is configured", str(path))
        # `find_spec` checks availability WITHOUT importing. A readiness probe must
        # not initialize a heavy framework: importing torch has side effects (CUDA
        # init among them) and readiness is called for experts that may never run.
        from importlib.util import find_spec

        for dependency in ("torch", "transformers"):
            try:
                found = find_spec(dependency) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                return (False, f"the {dependency!r} package is not installed", str(path))
        if self.vncorenlp_dir:
            from ..neural.runtime import probe_vncorenlp_capability

            capability = probe_vncorenlp_capability(self.vncorenlp_dir)
            if not capability.ready:
                reason = f"e3_vncorenlp_not_ready:{capability.reason_code or 'unknown'}"
                detail = capability.detail or str(Path(self.vncorenlp_dir))
                return (False, reason, detail)
        return (True, "", str(path))

    @property
    def loaded(self) -> bool:
        return self._expert is not None

    # -- lazy load -------------------------------------------------------------
    def prepare(self) -> None:
        """Build the model. Called only for an enabled, ready expert."""
        if self._expert is not None:
            return
        from ..neural.runtime import NeuralExpertConfig, build_segmenter, load_expert

        config = NeuralExpertConfig(
            checkpoint_path=self.checkpoint_path,
            expected_checkpoint_sha256=self.expected_checkpoint_sha256,
            pinned_model_revision=self.pinned_model_revision,
            hf_model_id=self.hf_model_id,
            max_sequence_length=self.max_sequence_length,
            decision_threshold=self.decision_threshold,
            batch_size=self.batch_size,
            model_cache_dir=self.model_cache_dir,
            vncorenlp_dir=self.vncorenlp_dir,
        )
        segmenter = build_segmenter(self.vncorenlp_dir) if self.vncorenlp_dir else None
        self._expert = load_expert(config, segmenter=segmenter)
        fingerprint = self._expert.initial_fingerprint
        self.checkpoint_sha256 = getattr(fingerprint, "sha256", self.expected_checkpoint_sha256)

    def verify_checkpoint_unchanged(self) -> None:
        """Re-hash the checkpoint and refuse if anything about it moved."""
        if self._expert is not None:
            self._expert.verify_checkpoint_unchanged()

    # -- inference -------------------------------------------------------------
    def propose(
        self,
        graph: DocumentGraph,
        routings: Sequence[NodeRouting],
    ) -> tuple[ExpertSpanProposal, ...]:
        """Decode this document and return verified proposals. Forward-only."""
        del routings  # route evidence is attached by the lattice, not here
        if self._expert is None:
            raise RuntimeError("E3 was not prepared; the registry calls prepare() before propose()")
        decoded = self._expert.predict_spans(
            [{"example_id": graph.document_id, "text": graph.original_text}]
        )
        spans = decoded.get(graph.document_id, ())
        proposals: list[ExpertSpanProposal] = []
        for ordinal, span in enumerate(spans, start=1):
            # Spec §4, re-verified here: the decoder already checked, and a second
            # check costs nothing against the cost of one wrong offset reaching L9.
            if graph.original_text[span.start : span.end] != span.text:
                raise RuntimeError(
                    f"E3 decoded [{span.start}, {span.end}) whose text does not "
                    "slice out of original_text (spec §4)"
                )
            proposals.append(
                ExpertSpanProposal(
                    document_id=graph.document_id,
                    start=span.start,
                    end=span.end,
                    text=span.text,
                    type_scores={span.entity_type: float(span.score)},
                    local_score=float(span.score),
                    expert_id=EXPERT_VIHEALTHBERT,
                    proposal_id=f"e3-{graph.document_id}-{ordinal:04d}",
                    original_start=span.start,
                    original_end=span.end,
                    features={
                        "neural_token_count": float(span.token_count),
                        "neural_mean_probability": float(span.score),
                    },
                    config_version=E3_EXPERT_CONTRACT_VERSION,
                    model_revision=self.model_revision,
                    checkpoint_sha256=self.checkpoint_sha256,
                )
            )
        return tuple(proposals)


def build_e3_expert(settings: Mapping[str, Any]) -> L3Expert:
    """Factory the registry calls. Reads settings; constructs no model."""
    return E3VietHealthBertExpert(
        checkpoint_path=str(settings.get("e3_checkpoint_path", DEFAULT_CHECKPOINT_PATH)),
        expected_checkpoint_sha256=str(
            settings.get("e3_expected_checkpoint_sha256", E3_CHECKPOINT_SHA256)
        ),
        pinned_model_revision=str(
            settings.get("e3_pinned_model_revision", E3_PINNED_MODEL_REVISION)
        ),
        hf_model_id=str(settings.get("e3_hf_model_id", E3_HF_MODEL_ID)),
        model_cache_dir=str(settings.get("e3_model_cache_dir", "")),
        vncorenlp_dir=str(settings.get("e3_vncorenlp_dir", "")),
        decision_threshold=float(settings.get("e3_decision_threshold", 0.5)),
        max_sequence_length=int(settings.get("e3_max_sequence_length", 256)),
        batch_size=int(settings.get("e3_batch_size", 8)),
    )


E3_REGISTRATION = ExpertRegistration(
    expert_id=EXPERT_VIHEALTHBERT,
    feature_flag=E3_FEATURE_FLAG,
    role=E3_ROLE,
    factory=build_e3_expert,
    checkpoint_key=E3_CHECKPOINT_KEY,
)

register(E3_REGISTRATION)


__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "E3_CHECKPOINT_KEY",
    "E3_CHECKPOINT_SHA256",
    "E3_EXPERT_CONTRACT_VERSION",
    "E3_FEATURE_FLAG",
    "E3_HF_MODEL_ID",
    "E3_PINNED_MODEL_REVISION",
    "E3_REGISTRATION",
    "E3_ROLE",
    "E3VietHealthBertExpert",
    "build_e3_expert",
]
