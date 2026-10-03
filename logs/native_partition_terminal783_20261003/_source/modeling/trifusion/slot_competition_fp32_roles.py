"""Keep role attention FP32 after the competitive M0 exposed missing Q/K gradients."""

import torch
import torch.nn.functional as F

from .slot_competition_roles import SlotCompetitionRoles, SlotCompetitionTriFusion, allocation_weights


class FP32SlotCompetitionRoles(SlotCompetitionRoles):
    def sample_context(self, patches, positions, context_queries, role):
        with torch.autocast(patches.device.type, enabled=False):
            patches, context_queries = patches.float(), context_queries.float()
            width = patches.shape[-1]
            queries = self.anchor_queries[None] + context_queries[:, role, None]
            queries = self.query_projections[role](F.layer_norm(queries, (width,)))
            keys = self.key_projections[role](F.layer_norm(patches, (width,)))
            values = self.value_projections[role](patches)
            scores = torch.matmul(queries[:, None], keys.transpose(-1, -2)) * width ** -0.5
            weights = allocation_weights(scores, self.attention_normalization)
            return self.output_projections[role](torch.matmul(weights, values))


class FP32SlotCompetitionTriFusion(SlotCompetitionTriFusion):
    def __init__(self, *args, attention_normalization, **kwargs):
        super().__init__(*args, attention_normalization=attention_normalization, **kwargs)
        with torch.random.fork_rng(devices=[]):
            roles = FP32SlotCompetitionRoles(
                attention_normalization=attention_normalization, grid=self.roles.grid,
                width=self.roles.width, anchor_side=self.roles.anchor_side,
                m1=True, m2=True, mamba_factory=kwargs["mamba_factory"],
            )
        roles.load_state_dict(self.roles.state_dict(), strict=True)
        self.roles = roles
