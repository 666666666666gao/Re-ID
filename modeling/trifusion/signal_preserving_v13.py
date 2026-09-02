"""Shared fusion and train-only statistics for TriFusion V13."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch
import torch.nn.functional as F


EXPERT_COUNT = 3
MODALITY_COUNT = 3
FIXED_ALPHA = 0.2


@dataclass(frozen=True, eq=False)
class V13FusionOutput:
    fused_embedding: torch.Tensor
    retrieval_embedding: torch.Tensor


@dataclass(frozen=True, eq=False)
class CounterfactualUtilityOutput:
    utilities: torch.Tensor
    full_margins: torch.Tensor
    reference_embedding_sha256_before: str
    reference_embedding_sha256_after: str


@dataclass(frozen=True, eq=False)
class ClusterBootstrapResult:
    observed_mean: float
    lower_bound: float
    cluster_count: int
    resamples: int


def compose_v13_fusion(
    baseline_embedding: torch.Tensor,
    modal_residual: torch.Tensor,
    weights: torch.Tensor,
) -> V13FusionOutput:
    """Compose the exact-prefix V13 embedding with fixed residual energy."""

    batch_size = baseline_embedding.shape[0]
    if modal_residual.shape[:3] != (batch_size, EXPERT_COUNT, MODALITY_COUNT):
        raise ValueError("modal_residual must have shape B,3,3,R")
    if weights.shape != (batch_size, EXPERT_COUNT, MODALITY_COUNT):
        raise ValueError("weights must have shape B,3,3")
    routed_bank = (modal_residual * weights[..., None]).flatten(1)
    suffix = (
        F.normalize(routed_bank, dim=1)
        * baseline_embedding.detach().norm(dim=1, keepdim=True)
        * FIXED_ALPHA
    )
    fused = torch.cat((baseline_embedding, suffix), dim=1)
    return V13FusionOutput(
        fused_embedding=fused,
        retrieval_embedding=F.normalize(fused, dim=1),
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _identity_margins(
    query_embedding: torch.Tensor,
    reference_embedding: torch.Tensor,
    identities: torch.Tensor,
    cameras: torch.Tensor,
) -> torch.Tensor:
    distances = torch.cdist(query_embedding, reference_embedding)
    margins = []
    for index, row in enumerate(distances):
        positives = (identities == identities[index]) & (cameras != cameras[index])
        negatives = identities != identities[index]
        if not bool(positives.any()) or not bool(negatives.any()):
            raise RuntimeError("identity margin requires cross-camera positives and negatives")
        margins.append(row[negatives].min() - row[positives].max())
    return torch.stack(margins)


def query_side_counterfactual_utilities(
    baseline_embedding: torch.Tensor,
    modal_residual: torch.Tensor,
    identities: torch.Tensor,
    cameras: torch.Tensor,
) -> CounterfactualUtilityOutput:
    """Measure each slot against one immutable uniform reference bank."""

    batch_size = baseline_embedding.shape[0]
    uniform = torch.full(
        (batch_size, EXPERT_COUNT, MODALITY_COUNT),
        1.0 / (EXPERT_COUNT * MODALITY_COUNT),
        dtype=baseline_embedding.dtype,
        device=baseline_embedding.device,
    )
    full = compose_v13_fusion(baseline_embedding, modal_residual, uniform)
    reference = full.retrieval_embedding.detach().clone()
    reference_sha_before = _tensor_sha256(reference)
    full_margins = _identity_margins(
        full.retrieval_embedding,
        reference,
        identities,
        cameras,
    )
    utilities = []
    for slot in range(EXPERT_COUNT * MODALITY_COUNT):
        removed = torch.full_like(uniform, 1.0 / 8.0)
        removed.flatten(1)[:, slot] = 0.0
        intervened = compose_v13_fusion(
            baseline_embedding,
            modal_residual,
            removed,
        )
        removed_margins = _identity_margins(
            intervened.retrieval_embedding,
            reference,
            identities,
            cameras,
        )
        utilities.append(full_margins - removed_margins)
    return CounterfactualUtilityOutput(
        utilities=torch.stack(utilities, dim=1).reshape(
            batch_size,
            EXPERT_COUNT,
            MODALITY_COUNT,
        ),
        full_margins=full_margins,
        reference_embedding_sha256_before=reference_sha_before,
        reference_embedding_sha256_after=_tensor_sha256(reference),
    )


def identity_cluster_bootstrap_lower_bound(
    differences: torch.Tensor,
    identities: torch.Tensor,
    *,
    seed: int,
    resamples: int,
) -> ClusterBootstrapResult:
    """Return the paired 95% lower bound while sampling whole identities."""

    values = differences.detach().cpu().numpy().astype(np.float64, copy=False)
    identity_values = identities.detach().cpu().numpy()
    clusters = np.unique(identity_values)
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    rows_by_cluster = {
        identity: np.flatnonzero(identity_values == identity) for identity in clusters
    }
    for index in range(resamples):
        sampled = generator.choice(clusters, size=len(clusters), replace=True)
        rows = np.concatenate([rows_by_cluster[identity] for identity in sampled])
        means[index] = values[rows].mean()
    return ClusterBootstrapResult(
        observed_mean=float(values.mean()),
        lower_bound=float(np.percentile(means, 2.5)),
        cluster_count=len(clusters),
        resamples=int(resamples),
    )


__all__ = [
    "ClusterBootstrapResult",
    "CounterfactualUtilityOutput",
    "FIXED_ALPHA",
    "V13FusionOutput",
    "compose_v13_fusion",
    "identity_cluster_bootstrap_lower_bound",
    "query_side_counterfactual_utilities",
]
