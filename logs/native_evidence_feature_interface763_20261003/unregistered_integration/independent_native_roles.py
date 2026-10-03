"""Unregistered integration draft; preserve semantic reading and add detail.

This source is intended for modeling/trifusion after the F3 foundation decision.
It has not been imported, deployed, source-reviewed, or trained.
"""
import torch
import torch.nn.functional as F

from .correspondence_roles import RoleEvidence
from .image_native_evidence import ImageNativeEvidenceReader
from .role_global_tokens import GlobalTokenRoles, GlobalTokenTriFusion
from .state import MODALITY_ORDER


class IndependentNativeRoles(GlobalTokenRoles):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.width == 128 and self.anchor_count == 16
        assert self.token_mode == 'static'
        self.detail_reader = ImageNativeEvidenceReader()

    def forward(self, stages, context, shared_global, images):
        context_queries = self.context_queries(context).reshape(context.shape[0], 3, self.width)
        cnn_input = self._depth_role(stages, 0)
        transformer_input = self._depth_role(stages, 1)
        mamba_input = self._depth_role(stages, 2)
        positions = self._positions(transformer_input[:, :, 0])
        height, grid_width = self.grid
        cnn_grid = cnn_input[:, :, 1:].reshape(-1, height, grid_width, self.width).permute(0, 3, 1, 2)
        cnn_grid = cnn_grid + self.cnn(cnn_grid)
        cnn_patches = cnn_grid.permute(0, 2, 3, 1).reshape(cnn_input.shape[0], 3, -1, self.width)
        # Preserve the original 128-token semantic sampler and its parameters.
        cnn = self.sample_context(cnn_patches, positions, context_queries, 0)
        detail_queries = self.anchor_queries[None] + context_queries[:, 0, None]
        detail_queries = detail_queries[:, None].expand(-1, 3, -1, -1)
        image_tensor = torch.stack([images[name] for name in MODALITY_ORDER], dim=1)
        cnn = cnn + self.detail_reader(image_tensor, detail_queries)
        transformer = self.sample_context(transformer_input[:, :, 1:], positions, context_queries, 1)
        mamba = self.sample_context(mamba_input[:, :, 1:], positions, context_queries, 2)
        queries = self.anchor_queries[None, None]
        cnn = self.output_norms[0](cnn + queries)
        transformer = transformer + queries + self.cnn_to_transformer(cnn)
        fixed = self.static_token_context[None, None].expand(context.shape[0], 3, -1)
        global_token = self.global_projection(fixed)
        sequence = torch.cat((global_token.unsqueeze(2), transformer), dim=2)
        transformer = self.transformer(sequence.flatten(0, 1))[:, 1:].reshape_as(transformer)
        transformer = self.output_norms[1](transformer)
        mamba_input = mamba + queries + self.transformer_to_mamba(transformer)
        sequence = mamba_input.permute(0, 2, 1, 3).reshape(-1, self.anchor_count * 3, self.width)
        sequence = self.mamba_norm(sequence)
        propagated = 0.5 * (self.mamba(sequence) + self.mamba(sequence.flip(1)).flip(1))
        mamba = self.output_norms[2](
            propagated.reshape(-1, self.anchor_count, 3, self.width).permute(0, 2, 1, 3))
        return RoleEvidence(cnn=cnn, transformer=transformer, mamba=mamba, positions=positions)


class RawFeatureSemanticTriFusion(GlobalTokenTriFusion):
    """Semantic control with an explicit feature-only training interface.

    The future trainer chooses matched heads and losses after F3 closes.
    `forward_features` does not run the historical normalized classifier.
    """
    def __init__(self, *args, **kwargs):
        assert kwargs['token_mode'] == 'static'
        super().__init__(*args, **kwargs)

    def role_evidence(self, batch, stages, context, shared_global):
        return self.roles(stages, context, shared_global)

    def forward_features(self, batch):
        stages, shared_global = self.backbone(batch['images'], batch['camera_ids'])
        context = F.normalize(shared_global.float().reshape(-1, 3, 512).mean(dim=1), dim=1)
        evidence = self.role_evidence(batch, stages, context, shared_global)
        correction = self.read_evidence(evidence)
        raw_fused = shared_global.float() + self.readout_gain * correction.float()
        fused = F.normalize(raw_fused, dim=1)
        return {'raw_fused': raw_fused, 'fused': fused,
                'shared_global': shared_global, 'correction': correction}

    def forward(self, batch, *, return_aux=False):
        output = self.forward_features(batch)
        if not return_aux:
            return output['fused']
        return {**output,
                'joint_local': F.normalize(output['correction'].float(), dim=1),
                'logits': self.classifier(self.neck(output['fused']))}


class IndependentNativeTriFusion(RawFeatureSemanticTriFusion):
    def __init__(self, *args, **kwargs):
        assert kwargs['token_mode'] == 'static'
        super().__init__(*args, **kwargs)
        # Added random initialization must not shift the original model's RNG.
        with torch.random.fork_rng(devices=[]):
            roles = IndependentNativeRoles(
                token_mode='static', grid=self.roles.grid, width=self.roles.width,
                anchor_side=self.roles.anchor_side, m1=True, m2=True,
                mamba_factory=kwargs['mamba_factory'])
        state = roles.state_dict()
        state.update(self.roles.state_dict())
        roles.load_state_dict(state, strict=True)
        self.roles = roles

    def role_evidence(self, batch, stages, context, shared_global):
        return self.roles(stages, context, shared_global, batch['images'])
