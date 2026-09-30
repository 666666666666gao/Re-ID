"""Full Patch memory with independent or competitive slot allocation."""

import torch
import torch.nn.functional as F

from .patch_memory_roles import PatchMemoryRoles, PatchMemoryTriFusion


def allocation_weights(scores, normalization):
    assert normalization in ("independent", "competitive")
    logits = scores.float()
    if normalization == "independent":
        return logits.softmax(dim=-1)
    weights = logits.softmax(dim=-2)
    return weights / weights.sum(dim=-1, keepdim=True)


class SlotCompetitionRoles(PatchMemoryRoles):
    def __init__(self, *, attention_normalization, **kwargs):
        super().__init__(memory_mode="full", **kwargs)
        assert attention_normalization in ("independent", "competitive")
        self.attention_normalization = attention_normalization

    def sample_context(self, patches, positions, context_queries, role):
        batch, modalities, tokens, width = patches.shape
        assert modalities == 3 and tokens == self.grid[0] * self.grid[1]
        queries = self.anchor_queries[None] + context_queries[:, role, None]
        queries = self.query_projections[role](F.layer_norm(queries, (width,)))
        keys = self.key_projections[role](F.layer_norm(patches, (width,)))
        values = self.value_projections[role](patches)
        scores = torch.matmul(queries[:, None], keys.transpose(-1, -2)) * width ** -0.5
        weights = allocation_weights(scores, self.attention_normalization).to(values.dtype)
        return self.output_projections[role](torch.matmul(weights, values))


class SlotCompetitionTriFusion(PatchMemoryTriFusion):
    def __init__(self, *args, attention_normalization, **kwargs):
        super().__init__(*args, memory_mode="full", **kwargs)
        with torch.random.fork_rng(devices=[]):
            roles = SlotCompetitionRoles(
                attention_normalization=attention_normalization, grid=self.roles.grid,
                width=self.roles.width, anchor_side=self.roles.anchor_side,
                m1=True, m2=True, mamba_factory=kwargs["mamba_factory"],
            )
        roles.load_state_dict(self.roles.state_dict(), strict=True)
        self.roles = roles
        self.attention_normalization = attention_normalization
