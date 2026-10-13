"""Describe cross-spectral changes relative to the same semantic reading.

Low-rank modality products follow the LMF factorization idea. The matched
reference follows this project's V8 comparison principle. Neither is an
identity-complementarity guarantee; additive and unreferenced controls are
required before attributing a retrieval gain to either part.
"""
import torch
from torch import nn
import torch.nn.functional as F


def semantic_reference(roles, stages, context):
    """Read the original semantic patches without CNN/T/M or their bridges."""
    stages, context = stages.detach(), context.detach()
    queries = roles.context_queries(context).reshape(context.shape[0], 3, roles.width)
    features = [roles._depth_role(stages, role) for role in range(3)]
    positions = roles._positions(features[1][:, :, 0])
    return tuple(roles.output_norms[role](roles.sample_context(
        feature[:, :, 1:], positions, queries, role) + roles.anchor_queries[None, None])
        for role, feature in enumerate(features))


class JointSpectralDelta(nn.Module):
    """Same-parameter additive/product controls at the actual evidence readout.

    Pool slots separately in each role/modality before combining modalities.
    No correspondence between equal-numbered cross-spectral slots is assumed.
    """
    MODES = ('add_delta', 'joint_delta', 'joint_absolute')

    def __init__(self, *, mode, width=128, rank=4):
        super().__init__()
        assert mode in self.MODES and width == 128 and rank == 4
        self.mode = mode
        self.width = width
        self.factors = nn.Parameter(torch.empty(3, 3, rank, width, width))
        for role in range(3):
            for modality in range(3):
                for component in range(rank):
                    nn.init.xavier_normal_(self.factors[role, modality, component])
        self.rank_weights = nn.Parameter(torch.full((3, rank), 1.0 / rank))

    def _describe(self, regions):
        assert len(regions) == 3
        assert all(value.shape[1:] == (3, 16, self.width) for value in regions)
        pooled = torch.stack(regions, dim=1).float().mean(dim=3)
        factors = torch.einsum('brmi,rmkio->brmko', pooled, self.factors.float())
        factors = F.layer_norm(factors, (self.width,))
        combined = factors.sum(dim=2) if self.mode == 'add_delta' else factors.prod(dim=2)
        descriptor = torch.einsum('brko,rk->bro', combined, self.rank_weights.float())
        return descriptor, factors

    def forward(self, processed, reference, readout_weight):
        assert readout_weight.shape == (1536, 3 * self.width)
        with torch.autocast(processed[0].device.type, enabled=False):
            actual, factors = self._describe(processed)
            baseline, reference_factors = self._describe(reference)
            delta = actual if self.mode == 'joint_absolute' else actual - baseline
            normalized = F.layer_norm(delta, (self.width,))
            correction = F.linear(normalized.flatten(1), readout_weight.float())
        return correction, {
            'actual_descriptor': actual,
            'reference_descriptor': baseline,
            'delta_descriptor': delta,
            'modality_factors': factors,
            'reference_factors': reference_factors,
        }
