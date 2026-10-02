"""Keep CNN semantic values, then test an additive image-native detail source."""
import torch
from torch import nn
import torch.nn.functional as F

from .native_detail_roles import NativeDetailRoles, NativeDetailTriFusion
from .role_global_tokens import GlobalTokenRoles, GlobalTokenTriFusion
from .state import MODALITY_ORDER


class SemanticNativeRoles(GlobalTokenRoles):
    def __init__(self, *, value_source, **kwargs):
        assert value_source in ('semantic', 'combined')
        super().__init__(**kwargs)
        self.value_source = value_source
        assert self.width == 128
        if value_source == 'combined':
            self.detail_stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            )
            assert sum(p.numel() for p in self.detail_stem.parameters()) == 93248

    def native_evidence(self, semantic_grid, images, context_queries):
        batch = context_queries.shape[0]
        size = (self.grid[0] * 2, self.grid[1] * 2)
        semantic = F.interpolate(semantic_grid, size=size, mode='bilinear', align_corners=False)
        values = semantic
        if self.value_source == 'combined':
            image = torch.stack([images[name] for name in MODALITY_ORDER], dim=1)
            image = image.flatten(0, 1)
            detail = self.detail_stem(image)
            assert detail.shape == semantic.shape
            values = semantic + detail
        with torch.autocast(semantic.device.type, enabled=False):
            semantic = semantic.float().flatten(2).transpose(1, 2).reshape(batch, 3, 512, self.width)
            values = values.float().flatten(2).transpose(1, 2).reshape_as(semantic)
            queries = self.anchor_queries[None] + context_queries[:, 0, None].float()
            queries = self.query_projections[0](F.layer_norm(queries, (self.width,)))
            keys = self.key_projections[0](F.layer_norm(semantic, (self.width,)))
            values = self.value_projections[0](values)
            weights = (queries[:, None] @ keys.transpose(-1, -2) * self.width ** -0.5).softmax(dim=-1)
            return self.output_projections[0](weights @ values)

    # Reuse the sealed role operator/bridge order; only CNN evidence values change.
    forward = NativeDetailRoles.forward


class SemanticNativeTriFusion(GlobalTokenTriFusion):
    def __init__(self, *args, value_source, **kwargs):
        assert kwargs['token_mode'] == 'static'
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            roles = SemanticNativeRoles(
                value_source=value_source, token_mode='static', grid=self.roles.grid,
                width=self.roles.width, anchor_side=self.roles.anchor_side,
                m1=True, m2=True, mamba_factory=kwargs['mamba_factory'])
        state = roles.state_dict()
        state.update(self.roles.state_dict())
        roles.load_state_dict(state, strict=True)
        self.roles = roles

    forward = NativeDetailTriFusion.forward
