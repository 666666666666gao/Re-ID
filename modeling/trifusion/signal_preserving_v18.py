"""Source-paired view-variation projection for the V18 residual representation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .signal_preserving_v17 import TriadicCorrectionOutputV17
from .state import EXPERT_ORDER


def estimate_paired_direction(features, identities, cameras):
    """Fit one principal within-identity cross-camera direction on source only."""
    differences = []
    pairs = []
    for identity in identities.unique(sorted=True):
        mask = identities == identity
        views = cameras[mask].unique(sorted=True)
        for i, camera_a in enumerate(views):
            for camera_b in views[i + 1:]:
                first = features[mask & (cameras == camera_a)].double().mean(dim=0)
                second = features[mask & (cameras == camera_b)].double().mean(dim=0)
                differences.append(second - first)
                pairs.append([int(identity), int(camera_a), int(camera_b)])
    differences = torch.stack(differences)
    _u, singular, vh = torch.linalg.svd(differences, full_matrices=False)
    assert float(singular[0]) > 0
    direction = vh[0].float()
    direction = direction * direction[direction.abs().argmax()].sign()
    return direction, {
        "identity_camera_pairs": pairs,
        "pair_difference_vectors": differences.float().tolist(),
        "singular_values": singular.tolist(),
        "top_direction_energy_fraction": float(singular[0].square() / singular.square().sum()),
    }


class PairedViewProjectionV18(nn.Module):
    """Keep corrected expert embeddings orthogonal to a source-fitted direction."""

    def __init__(self, correction, directions, *, enabled):
        super().__init__()
        self.core = correction
        self.register_buffer("directions", directions.detach().clone())
        self.enabled = bool(enabled)
        assert directions.shape[0] == len(EXPERT_ORDER)
        assert torch.allclose(directions.norm(dim=1), torch.ones(len(EXPERT_ORDER), device=directions.device))

    def project(self, value, index):
        direction = self.directions[index].to(dtype=value.dtype)
        return F.normalize(value - (value * direction).sum(dim=1, keepdim=True) * direction, dim=1)

    def forward(self, residual_embeddings):
        if not self.enabled:
            return self.core(residual_embeddings)
        inputs = {expert: self.project(residual_embeddings[expert], i) for i, expert in enumerate(EXPERT_ORDER)}
        output = self.core(inputs)
        corrected = {expert: self.project(output.corrected_residuals[expert], i) for i, expert in enumerate(EXPERT_ORDER)}
        return TriadicCorrectionOutputV17(
            corrected_residuals=corrected,
            fused_residual=F.normalize(torch.cat(tuple(corrected.values()), dim=1), dim=1),
        )
