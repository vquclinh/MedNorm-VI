"""Model registry and deployed-parameter budget for GraphCENT (0080).

Every learned component that runs at inference is declared here with its repo, pinned
revision, licence, pooling contract and role. Two things are deliberately NOT hard-coded:
the parameter counts, and the resolved revisions. Both are filled in at Colab runtime from
the actual checkpoints, because a hard-coded count is an assumption, not evidence.

Pooling is part of the registry because it is not interchangeable. Cross-lingual SapBERT is
trained to be read at **CLS before the pooler**; mean-pooling it silently degrades it. The
BioBERT sentence model is a SentenceTransformer with configured **mean** pooling; forcing CLS
on it is the same mistake in reverse. `RetrieverSpec.pooling` is what the encoder obeys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PARAMETER_CAP = 9_000_000_000

#: A Hugging Face commit SHA. A branch name ("main") is not a revision: it moves, so a
#: manifest recording one proves nothing about which weights ran.
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

#: Local checkpoints have no Hugging Face revision. Their identity is the file digest, and
#: it is labelled as such so nobody mistakes it for a hub commit.
LOCAL_DIGEST_PREFIX = "sha256:"
_LOCAL_SHA256 = re.compile(rf"^{LOCAL_DIGEST_PREFIX}[0-9a-f]{{64}}$")


def is_hub_revision(value: str) -> bool:
    """True only for an immutable Hugging Face commit SHA."""
    return bool(_COMMIT_SHA.match((value or "").strip()))


def is_local_checkpoint_revision(value: str) -> bool:
    """True only for a labelled local checkpoint SHA256 digest."""
    return bool(_LOCAL_SHA256.match((value or "").strip()))


def is_pinned_revision(value: str) -> bool:
    """True for an immutable identity: a 40-hex hub commit, or a labelled file digest."""
    text = (value or "").strip()
    return is_hub_revision(text) or is_local_checkpoint_revision(text)

POOLING_CLS = "cls"
POOLING_MEAN = "mean"
POOLINGS: tuple[str, ...] = (POOLING_CLS, POOLING_MEAN)

ROLE_DIAGNOSIS = "diagnosis"
ROLE_DRUG = "drug"

#: Licences whose terms restrict commercial use. Recorded faithfully as the model card
#: declares them; whether a restriction matters for this competition is the owner's call,
#: which is why an enabled model with one of these fails closed until explicitly accepted.
#: This is a routing rule, not a legal conclusion.
LICENCE_REVIEW_REQUIRED: frozenset[str] = frozenset({
    "cc-by-nc-3.0", "cc-by-nc-4.0", "cc-by-nc-sa-3.0", "cc-by-nc-sa-4.0",
    "unspecified-see-model-card",
})


@dataclass(frozen=True, slots=True)
class RetrieverSpec:
    """One frozen pretrained retriever. `revision` and `parameter_count` resolve at runtime."""

    key: str
    repo_id: str
    licence: str
    licence_source: str
    pooling: str
    role: tuple[str, ...]
    intended_domain: str
    expected_disk_gib: float
    revision: str = ""
    parameter_count: int | None = None
    embedding_dim: int | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.pooling not in POOLINGS:
            raise ValueError(f"{self.key}: unknown pooling {self.pooling!r}")

    @property
    def licence_needs_review(self) -> bool:
        return self.licence.lower() in LICENCE_REVIEW_REQUIRED

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "licence": self.licence,
            "licence_source": self.licence_source,
            "licence_needs_review": self.licence_needs_review,
            "pooling": self.pooling,
            "role": list(self.role),
            "intended_domain": self.intended_domain,
            "parameter_count": self.parameter_count,
            "embedding_dim": self.embedding_dim,
            "expected_disk_gib": self.expected_disk_gib,
            "enabled": self.enabled,
        }


#: The 0080 retriever stack. KRISSBERT is intentionally absent - the budget headroom is
#: needed and a fourth retriever can be added once the first run justifies it.
DEFAULT_RETRIEVERS: tuple[RetrieverSpec, ...] = (
    RetrieverSpec(
        key="sapbert_xlmr",
        repo_id="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
        licence="mit",  # cambridgeltl SapBERT repository declares MIT
        licence_source="https://huggingface.co/cambridgeltl/"
        "SapBERT-UMLS-2020AB-all-lang-from-XLMR (model card) + cambridgeltl/sapbert (code)",
        pooling=POOLING_CLS,  # official cross-lingual SapBERT extraction
        role=(ROLE_DIAGNOSIS, ROLE_DRUG),
        intended_domain="multilingual UMLS synonym alignment",
        expected_disk_gib=1.1,
    ),
    RetrieverSpec(
        key="biobert_mnli",
        repo_id="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        licence="cc-by-nc-3.0",  # as declared on the model card: NON-COMMERCIAL
        licence_source="https://huggingface.co/pritamdeka/"
        "BioBERT-mnli-snli-scinli-scitail-mednli-stsb (model card)",
        pooling=POOLING_MEAN,  # SentenceTransformer configured pooling
        role=(ROLE_DIAGNOSIS, ROLE_DRUG),
        intended_domain="clinical sentence similarity",
        expected_disk_gib=0.44,
        # Disabled by DEFAULT. The code is complete and supported; enabling it is an
        # explicit decision recorded in the profile, never a silent default.
        enabled=False,
    ),
    RetrieverSpec(
        key="clinlinker_kb_gp",
        repo_id="ICB-UMA/ClinLinker-KB-GP",
        licence="apache-2.0",  # as declared on the current model card
        licence_source="https://huggingface.co/ICB-UMA/ClinLinker-KB-GP (model card)",
        pooling=POOLING_CLS,
        role=(ROLE_DIAGNOSIS,),  # Spanish clinical concepts; weak evidence for medications
        intended_domain="ontology-enriched clinical entity linking",
        expected_disk_gib=0.44,
    ),
)


@dataclass(frozen=True, slots=True)
class DeployedModel:
    """Any learned component counted against the 9B cap."""

    name: str
    parameter_count: int
    revision: str = ""
    licence: str = ""
    role: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": self.parameter_count,
            "revision": self.revision,
            "licence": self.licence,
            "role": self.role,
        }


class ParameterBudgetExceeded(RuntimeError):
    """Raised when the summed deployed parameters reach the cap. Never downgraded."""


class LicenceReviewRequired(RuntimeError):
    """Raised when an enabled model's licence has not been explicitly cleared."""


class RevisionNotPinned(RuntimeError):
    """Raised when a deployed model cannot prove which exact weights it is.

    Not a warning. A run whose provenance is a floating branch cannot be reproduced, and
    a manifest that records `""` or `"main"` is worse than no manifest because it looks
    like evidence.
    """


@dataclass
class ModelManifest:
    """Resolved stack. Built at runtime; refuses to certify what it has not measured."""

    deployed: list[DeployedModel] = field(default_factory=list)
    retrievers: list[RetrieverSpec] = field(default_factory=list)
    resolved_revisions: dict[str, str] = field(default_factory=dict)
    licence_overrides_accepted: tuple[str, ...] = ()

    @property
    def total_parameters(self) -> int:
        return sum(m.parameter_count for m in self.deployed)

    def assert_within_cap(self) -> int:
        unmeasured = [m.name for m in self.deployed if m.parameter_count <= 0]
        if unmeasured:
            raise ParameterBudgetExceeded(
                f"cannot certify the budget: parameter count not measured for {unmeasured}. "
                "Counts must come from the loaded checkpoints, never from a constant."
            )
        total = self.total_parameters
        if total >= PARAMETER_CAP:
            raise ParameterBudgetExceeded(
                f"deployed parameters {total:,} >= cap {PARAMETER_CAP:,}. "
                "Disable a retriever in the profile or drop a component."
            )
        return total

    def assert_licences_cleared(self) -> None:
        blocked = [
            spec.repo_id
            for spec in self.retrievers
            if spec.enabled
            and spec.licence_needs_review
            and spec.key not in self.licence_overrides_accepted
        ]
        if blocked:
            raise LicenceReviewRequired(
                f"these models carry a licence needing explicit review: {blocked}. "
                "Either disable them in the profile (`retrievers.<key>.enabled: false`) or "
                "record the cleared decision in `licence_overrides_accepted`. A licence is "
                "never silently ignored."
            )

    def assert_revisions_pinned(self) -> None:
        """Every enabled retriever and every deployed component names exact weights.

        Called before the manifest is written, so a manifest that exists on disk is one
        whose run can be reproduced. Disabled retrievers are exempt: they were never
        acquired, and inventing a revision for them would be the opposite of provenance.
        """
        blocked = [
            f"{spec.key} ({spec.revision or 'empty'})"
            for spec in self.retrievers
            if spec.enabled and not is_hub_revision(spec.revision)
        ]
        if self.resolved_revisions:
            blocked += [
                f"resolved_revisions.{spec.key} "
                f"({self.resolved_revisions.get(spec.key) or 'missing'})"
                for spec in self.retrievers
                if spec.enabled and self.resolved_revisions.get(spec.key) != spec.revision
            ]
        deployed_by_name = {model.name: model for model in self.deployed}
        blocked += [
            f"deployed {spec.key} "
            f"({deployed_by_name[spec.repo_id].revision or 'empty'})"
            for spec in self.retrievers
            if spec.enabled
            and spec.repo_id in deployed_by_name
            and deployed_by_name[spec.repo_id].revision != spec.revision
        ]
        blocked += [
            f"{model.name} ({model.revision or 'empty'})"
            for model in self.deployed
            if (
                not is_local_checkpoint_revision(model.revision)
                if model.name == "ViHealthBERT E3"
                else not is_hub_revision(model.revision)
            )
        ]
        if blocked:
            raise RevisionNotPinned(
                f"these deployed models have no immutable revision: {blocked}. "
                "Resolve the exact commit SHA before acquisition for hub models, and "
                "record a labelled sha256 digest for local checkpoints. A branch name or "
                "an empty string cannot identify the weights that produced a submission."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "deployed": [m.as_dict() for m in self.deployed],
            "retrievers": [r.as_dict() for r in self.retrievers],
            "resolved_revisions": dict(sorted(self.resolved_revisions.items())),
            "total_deployed_parameters": self.total_parameters,
            "parameter_cap": PARAMETER_CAP,
            "under_cap": self.total_parameters < PARAMETER_CAP,
            "licence_overrides_accepted": list(self.licence_overrides_accepted),
            "training_performed": False,
        }


def retrievers_for(specs: tuple[RetrieverSpec, ...], role: str) -> tuple[RetrieverSpec, ...]:
    """Enabled retrievers declared for this entity role, in registry order."""
    return tuple(s for s in specs if s.enabled and role in s.role)


__all__ = [
    "DEFAULT_RETRIEVERS",
    "LICENCE_REVIEW_REQUIRED",
    "LOCAL_DIGEST_PREFIX",
    "PARAMETER_CAP",
    "POOLINGS",
    "POOLING_CLS",
    "POOLING_MEAN",
    "ROLE_DIAGNOSIS",
    "ROLE_DRUG",
    "DeployedModel",
    "LicenceReviewRequired",
    "ModelManifest",
    "ParameterBudgetExceeded",
    "RetrieverSpec",
    "RevisionNotPinned",
    "is_hub_revision",
    "is_local_checkpoint_revision",
    "is_pinned_revision",
    "retrievers_for",
]
