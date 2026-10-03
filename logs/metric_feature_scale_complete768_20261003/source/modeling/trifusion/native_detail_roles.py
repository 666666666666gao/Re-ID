"""Change CNN attention values to image-native detail; retain semantic keys and readout."""
import torch
from torch import nn
import torch.nn.functional as F

from .correspondence_roles import RoleEvidence
from .role_global_tokens import GlobalTokenRoles, GlobalTokenTriFusion
from .state import MODALITY_ORDER


class NativeDetailRoles(GlobalTokenRoles):
    def __init__(self, *, detail_resolution, **kwargs):
        assert detail_resolution in ('high', 'low')
        super().__init__(**kwargs)
        self.detail_resolution = detail_resolution
        self.detail_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
        )
        assert self.width == 128
        assert sum(p.numel() for p in self.detail_stem.parameters()) == 93248

    def native_evidence(self, semantic_grid, images, context_queries):
        # Both conditions expose 512 candidates. Upsampling the low condition
        # aligns the interface; it cannot recreate the removed input detail.
        image = torch.stack([images[name] for name in MODALITY_ORDER], dim=1)
        batch, modalities, channels, height, width = image.shape
        image = image.reshape(batch * modalities, channels, height, width)
        if self.detail_resolution == 'low':
            image = F.avg_pool2d(image, 2)
        size = (self.grid[0] * 2, self.grid[1] * 2)
        detail = self.detail_stem(image)
        if self.detail_resolution == 'low':
            detail = F.interpolate(detail, size=size, mode='bilinear', align_corners=False)
        assert detail.shape[-2:] == size
        semantic = F.interpolate(semantic_grid, size=size, mode='bilinear', align_corners=False)
        with torch.autocast(detail.device.type, enabled=False):
            detail = detail.float().flatten(2).transpose(1, 2).reshape(batch, 3, 512, self.width)
            semantic = semantic.float().flatten(2).transpose(1, 2).reshape_as(detail)
            queries = self.anchor_queries[None] + context_queries[:, 0, None].float()
            queries = self.query_projections[0](F.layer_norm(queries, (self.width,)))
            keys = self.key_projections[0](F.layer_norm(semantic, (self.width,)))
            values = self.value_projections[0](detail)
            weights = (queries[:, None] @ keys.transpose(-1, -2) * self.width ** -0.5).softmax(dim=-1)
            return self.output_projections[0](weights @ values)

    def forward(self, stages, context, shared_global, images):
        context_queries = self.context_queries(context).reshape(context.shape[0], 3, self.width)
        cnn_input = self._depth_role(stages, 0)
        transformer_input = self._depth_role(stages, 1)
        mamba_input = self._depth_role(stages, 2)
        positions = self._positions(transformer_input[:, :, 0])
        height, width = self.grid
        cnn_grid = cnn_input[:, :, 1:].reshape(-1, height, width, self.width).permute(0, 3, 1, 2)
        cnn_grid = cnn_grid + self.cnn(cnn_grid)
        cnn = self.native_evidence(cnn_grid, images, context_queries)
        transformer = self.sample_context(transformer_input[:, :, 1:], positions, context_queries, 1)
        mamba = self.sample_context(mamba_input[:, :, 1:], positions, context_queries, 2)
        queries = self.anchor_queries[None, None]
        cnn = self.output_norms[0](cnn + queries)
        transformer = transformer + queries + self.cnn_to_transformer(cnn)
        fixed = self.static_token_context[None, None].expand(context.shape[0], 3, -1)
        sequence = torch.cat((self.global_projection(fixed).unsqueeze(2), transformer), dim=2)
        transformer = self.transformer(sequence.flatten(0, 1))[:, 1:].reshape_as(transformer)
        transformer = self.output_norms[1](transformer)
        sequence = (mamba + queries + self.transformer_to_mamba(transformer))
        sequence = sequence.permute(0, 2, 1, 3).reshape(-1, self.anchor_count * 3, self.width)
        sequence = self.mamba_norm(sequence)
        propagated = 0.5 * (self.mamba(sequence) + self.mamba(sequence.flip(1)).flip(1))
        mamba = self.output_norms[2](
            propagated.reshape(-1, self.anchor_count, 3, self.width).permute(0, 2, 1, 3))
        return RoleEvidence(cnn=cnn, transformer=transformer, mamba=mamba, positions=positions)


class NativeDetailTriFusion(GlobalTokenTriFusion):
    def __init__(self, *args, detail_resolution, **kwargs):
        assert kwargs['token_mode'] == 'static'
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            roles = NativeDetailRoles(
                detail_resolution=detail_resolution, token_mode='static', grid=self.roles.grid,
                width=self.roles.width, anchor_side=self.roles.anchor_side,
                m1=True, m2=True, mamba_factory=kwargs['mamba_factory'])
        state = roles.state_dict()
        state.update(self.roles.state_dict())
        roles.load_state_dict(state, strict=True)
        self.roles = roles

    def forward(self, batch, *, return_aux=False):
        stages, shared = self.backbone(batch['images'], batch['camera_ids'])
        context = F.normalize(shared.float().reshape(-1, 3, 512).mean(dim=1), dim=1)
        evidence = self.roles(stages, context, shared, batch['images'])
        correction = self.read_evidence(evidence)
        fused = F.normalize(shared.float() + self.readout_gain * correction.float(), dim=1)
        if not return_aux:
            return fused
        return {'fused': fused, 'shared_global': shared,
                'joint_local': F.normalize(correction.float(), dim=1),
                'logits': self.classifier(self.neck(fused))}
