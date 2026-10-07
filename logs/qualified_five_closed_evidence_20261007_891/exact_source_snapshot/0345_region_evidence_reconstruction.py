"""Visual candidate reconstruction before the existing CNN-region reader."""
import torch
from torch import nn

from .role_global_tokens import GlobalTokenRoles
from .role_input_detach import DetachedSemanticTriFusion


class RegionCandidateReconstruction(nn.Module):
    def __init__(self, query_mode):
        super().__init__()
        assert query_mode in ('patch', 'mean')
        self.query_mode = query_mode
        self.anchors = nn.Parameter(torch.randn(16, 128) * 0.02)
        self.anchor_offsets = nn.Sequential(nn.Linear(128, 16), nn.GELU(),
                                            nn.Linear(16, 16 * 128))
        self.query_norm = nn.LayerNorm(128)
        self.anchor_norm = nn.LayerNorm(128)
        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(128, 128, bias=False)
        self.value = nn.Linear(128, 128)
        self.output = nn.Linear(128, 128, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, patches):
        assert patches.shape[-2:] == (128, 128)
        with torch.autocast(patches.device.type, enabled=False):
            patches = patches.float()
            pooled = patches.mean(dim=-2)
            offsets = self.anchor_offsets(pooled).reshape(*pooled.shape[:-1], 16, 128)
            anchors = self.anchor_norm(self.anchors + offsets)
            query_source = (patches if self.query_mode == 'patch'
                            else pooled.unsqueeze(-2).expand_as(patches))
            query = self.query(self.query_norm(query_source))
            key, value = self.key(anchors), self.value(anchors)
            weights = (query @ key.transpose(-1, -2) * 128 ** -0.5).softmax(dim=-1)
            return patches + self.output(weights @ value)


class ReconstructedSemanticRoles(GlobalTokenRoles):
    def __init__(self, *, reconstruction_query_mode, **kwargs):
        super().__init__(**kwargs)
        self.reconstruction = RegionCandidateReconstruction(reconstruction_query_mode)

    def sample_context(self, patches, positions, context_queries, role):
        if role == 0:
            patches = self.reconstruction(patches)
        return super().sample_context(patches, positions, context_queries, role)


class ReconstructedSemanticTriFusion(DetachedSemanticTriFusion):
    def __init__(self, *args, reconstruction_query_mode, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            roles = ReconstructedSemanticRoles(
                reconstruction_query_mode=reconstruction_query_mode,
                token_mode='static', grid=self.roles.grid, width=self.roles.width,
                anchor_side=self.roles.anchor_side, m1=True, m2=True,
                mamba_factory=kwargs['mamba_factory'])
        state = roles.state_dict()
        state.update(self.roles.state_dict())
        roles.load_state_dict(state, strict=True)
        self.roles = roles


class PatchReconstructionTriFusion(ReconstructedSemanticTriFusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reconstruction_query_mode='patch', **kwargs)


class MeanReconstructionTriFusion(ReconstructedSemanticTriFusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reconstruction_query_mode='mean', **kwargs)
