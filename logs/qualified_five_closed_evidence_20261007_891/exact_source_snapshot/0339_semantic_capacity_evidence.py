"""Active semantic-source control for the independently read native branch."""
import torch
from torch import nn
import torch.nn.functional as F

from .image_native_evidence import ImageNativeEvidenceReader
from .independent_native_roles import IndependentNativeRoles, RawFeatureSemanticTriFusion
from .correspondence_roles import RoleEvidence


class SemanticCapacityReader(ImageNativeEvidenceReader):
    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        # 93,048 active parameters, versus the native CNN stem's 93,248.
        self.stem = nn.Sequential(nn.Linear(128, 202), nn.GELU(),
                                  nn.Linear(202, 202), nn.GELU(), nn.Linear(202, 128))

    def forward(self, patches, queries):
        batch, modalities, count, width = patches.shape
        height, grid_width = self.grid
        assert modalities == 3 and count == height * grid_width == 128 and width == 128
        detail = self.stem(patches).reshape(batch * modalities, height, grid_width, width)
        detail = F.interpolate(detail.permute(0, 3, 1, 2), size=(height * 2, grid_width * 2),
                               mode='bilinear', align_corners=False)
        detail = detail.flatten(2).transpose(1, 2).reshape(batch, modalities, 512, width)
        with torch.autocast(device_type=patches.device.type, enabled=False):
            detail = detail.float()
            query = self.query(F.layer_norm(queries.float(), (width,)))
            key = self.key(F.layer_norm(detail, (width,)))
            value = self.value(detail)
            weights = (query @ key.transpose(-1, -2) * width ** -0.5).softmax(dim=-1)
            return self.output(self.norm(weights @ value))


class SemanticCapacityRoles(IndependentNativeRoles):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        reader = SemanticCapacityReader(self.grid)
        state = reader.state_dict()
        state.update({key: value for key, value in self.detail_reader.state_dict().items()
                      if not key.startswith('stem.')})
        reader.load_state_dict(state, strict=True)
        self.detail_reader = reader

    def forward(self, stages, context, shared_global):
        context_queries = self.context_queries(context).reshape(context.shape[0], 3, self.width)
        cnn_input = self._depth_role(stages, 0)
        transformer_input = self._depth_role(stages, 1)
        mamba_input = self._depth_role(stages, 2)
        positions = self._positions(transformer_input[:, :, 0])
        height, grid_width = self.grid
        cnn_grid = cnn_input[:, :, 1:].reshape(-1, height, grid_width, self.width).permute(0, 3, 1, 2)
        cnn_grid = cnn_grid + self.cnn(cnn_grid)
        cnn_patches = cnn_grid.permute(0, 2, 3, 1).reshape(cnn_input.shape[0], 3, -1, self.width)
        cnn = self.sample_context(cnn_patches, positions, context_queries, 0)
        detail_queries = self.anchor_queries[None] + context_queries[:, 0, None]
        detail_queries = detail_queries[:, None].expand(-1, 3, -1, -1)
        # Same added Q/K/V read and residual location; only its source changes.
        cnn = cnn + self.detail_reader(cnn_patches, detail_queries)
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


class SemanticCapacityTriFusion(RawFeatureSemanticTriFusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            roles = SemanticCapacityRoles(token_mode='static', grid=self.roles.grid,
                width=self.roles.width, anchor_side=self.roles.anchor_side,
                m1=True, m2=True, mamba_factory=kwargs['mamba_factory'])
        state = roles.state_dict()
        state.update(self.roles.state_dict())
        roles.load_state_dict(state, strict=True)
        self.roles = roles

    def role_evidence(self, batch, stages, context, shared_global):
        return self.roles(stages.detach(), context.detach(), shared_global.detach())
