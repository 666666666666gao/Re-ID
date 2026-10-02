"""Separate prediction addresses from the role feature prediction heads."""

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from .correspondence_roles import CorrespondenceRoles, CorrespondenceTriFusion, RoleEvidence


class AddressedPredictionRoles(CorrespondenceRoles):
    def forward(self, stages: torch.Tensor, *, positions: torch.Tensor | None = None):
        if positions is None:
            return super().forward(stages)
        assert stages.ndim == 6 and stages.shape[:2] == (3, 3) and stages.shape[3] == 3
        assert positions.shape == (stages.shape[2], 3, self.anchor_count, 2)
        # The teacher's detached addresses replace only the masked prediction sampling.
        cnn_input = self._depth_role(stages, 0)
        transformer_input = self._depth_role(stages, 1)
        mamba_input = self._depth_role(stages, 2)
        height, grid_width = self.grid
        cnn_grid = cnn_input[:, :, 1:].reshape(-1, height, grid_width, self.width).permute(0, 3, 1, 2)
        cnn_grid = cnn_grid + self.cnn(cnn_grid)
        cnn = self._sample(cnn_grid.permute(0, 2, 3, 1).reshape(cnn_input.shape[0], 3, -1, self.width), positions)
        transformer = self._sample(transformer_input[:, :, 1:], positions)
        mamba = self._sample(mamba_input[:, :, 1:], positions)
        queries = self.anchor_queries[None, None]
        cnn = self.output_norms[0](cnn + queries)
        transformer_input = transformer + queries
        if self.m2:
            transformer_input = transformer_input + self.cnn_to_transformer(cnn)
        transformer = self.transformer(transformer_input.flatten(0, 1)).reshape_as(transformer)
        transformer = self.output_norms[1](transformer)
        mamba_input = mamba + queries
        if self.m2:
            mamba_input = mamba_input + self.transformer_to_mamba(transformer)
        sequence = mamba_input.permute(0, 2, 1, 3).reshape(mamba.shape[0], -1, self.width)
        sequence = self.mamba_norm(sequence)
        propagated = 0.5 * (self.mamba(sequence) + self.mamba(sequence.flip(1)).flip(1))
        mamba = self.output_norms[2](
            propagated.reshape(mamba.shape[0], self.anchor_count, 3, self.width).permute(0, 2, 1, 3)
        )
        return RoleEvidence(cnn=cnn, transformer=transformer, mamba=mamba, positions=positions)


class RoleFeaturePredictor(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, evidence: torch.Tensor, target_positions: torch.Tensor):
        # Target coordinates are training-only teacher guidance, not identity labels.
        return self.projection(torch.cat((self.norm(evidence), target_positions), dim=-1))


class RolePredictionTriFusion(CorrespondenceTriFusion):
    def __init__(self, *args, address_mode: str, prediction_mode: str, **kwargs):
        super().__init__(*args, **kwargs)
        assert address_mode in ('own', 'matched') and prediction_mode in ('direct', 'predictor')
        assert self.m3
        self.address_mode = address_mode
        self.prediction_mode = prediction_mode
        # Common role weights and subsequent masks retain the original initialization/RNG.
        with torch.random.fork_rng(devices=[]):
            roles = AddressedPredictionRoles(
                grid=self.roles.grid, width=self.roles.width, anchor_side=self.roles.anchor_side,
                m1=self.roles.m1, m2=self.roles.m2, mamba_factory=kwargs['mamba_factory'],
            )
            predictors = nn.ModuleList(RoleFeaturePredictor(self.roles.width) for _ in range(3))
        roles.load_state_dict(self.roles.state_dict(), strict=True)
        self.roles = roles
        self.teacher = deepcopy(roles)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.predictors = predictors
        if prediction_mode == 'direct':
            for parameter in self.predictors.parameters():
                parameter.requires_grad_(False)

    def _prediction_losses(self, images: dict[str, torch.Tensor], camera_ids: torch.Tensor):
        if self.address_mode == 'own' and self.prediction_mode == 'direct':
            return super()._prediction_losses(images, camera_ids)
        with torch.no_grad():
            clean_teacher, _ = self.backbone(images, camera_ids, adapter_bank=self.teacher_adapters)
            target = self.teacher(clean_teacher)
        masked_images, local_mask, missing = self._masked_images(images)
        masked_stages, _ = self.backbone(masked_images, camera_ids)
        target_positions = target.positions.detach()
        positions = target_positions if self.address_mode == 'matched' else None
        predicted = self.roles(masked_stages, positions=positions)
        features = (predicted.cnn, predicted.transformer, predicted.mamba)
        if self.prediction_mode == 'predictor':
            features = tuple(head(feature, target_positions) for head, feature in zip(self.predictors, features, strict=True))
        cnn, transformer, mamba = features
        batch_size = clean_teacher.shape[2]
        mask_at_anchor = F.grid_sample(
            local_mask.float().reshape(batch_size * 3, 1, *self.roles.grid),
            predicted.positions.reshape(batch_size * 3, self.roles.anchor_count, 1, 2),
            mode='nearest', align_corners=True,
        ).reshape(batch_size, 3, self.roles.anchor_count)
        local_error = (cnn.float() - target.cnn.float()).square().mean(dim=-1)
        local = (local_error * mask_at_anchor).sum() / mask_at_anchor.sum().clamp_min(1)
        target_rel = F.normalize(target.transformer.float(), dim=-1)
        predicted_rel = F.normalize(transformer.float(), dim=-1)
        relation = F.mse_loss(predicted_rel @ predicted_rel.transpose(-1, -2),
                              target_rel @ target_rel.transpose(-1, -2))
        rows = torch.arange(batch_size, device=missing.device)
        cross_modal = F.mse_loss(mamba[rows, missing].float(), target.mamba[rows, missing].float())
        return {'local': local, 'relation': relation, 'cross_modal': cross_modal}
