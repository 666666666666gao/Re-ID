"""Cross-layer, anchor-aligned CNN/Transformer/Mamba collaboration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .state import MODALITY_ORDER


@dataclass
class RoleEvidence:
    cnn: torch.Tensor
    transformer: torch.Tensor
    mamba: torch.Tensor
    positions: torch.Tensor


class CrossLayerAdaptedCLIP(nn.Module):
    """Inject role-conditioned adapters within a frozen CLIP visual path."""

    def __init__(self, signal: nn.Module, *, layers: tuple[int, int, int] = (3, 7, 11),
                 enabled: bool = True):
        super().__init__()
        self.signal = signal
        self.layers = layers
        self.enabled = enabled
        self.adapters = nn.ModuleList(
            nn.ModuleList(
                nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 64), nn.GELU(),
                              nn.Linear(64, 768)) for _ in range(3)
            ) for _ in layers
        )
        for depth in self.adapters:
            for adapter in depth:
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
        if not enabled:
            for parameter in self.adapters.parameters():
                parameter.requires_grad_(False)
        blocks = signal.clip_vision_encoder.base.transformer.resblocks
        assert len(blocks) == 12 and all(0 <= index < len(blocks) for index in layers)
        assert not hasattr(signal, "SIM") and not hasattr(signal, "AlignM")
        for parameter in signal.parameters():
            parameter.requires_grad_(False)
        signal.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.signal.eval()
        return self

    def forward(self, images: dict[str, torch.Tensor], camera_ids: torch.Tensor,
                adapter_bank: nn.ModuleList | None = None):
        bank = self.adapters if adapter_bank is None else adapter_bank
        captured: dict[int, list[torch.Tensor]] = {index: [] for index in self.layers}
        handles = []
        for stage_index, index in enumerate(self.layers):
            def capture(_module, _inputs, output, *, layer=index, stage=stage_index):
                if not self.enabled:
                    captured[layer].append(torch.stack([output.permute(1, 0, 2)] * 3, dim=0))
                    return output
                deltas = [adapter(output) for adapter in bank[stage]]
                captured[layer].append(torch.stack(
                    [(output + delta).permute(1, 0, 2) for delta in deltas], dim=0
                ))
                return output + sum(deltas) / 3
            handles.append(self.signal.clip_vision_encoder.base.transformer.resblocks[index].register_forward_hook(capture))
        globals_by_modal = []
        try:
            for name in MODALITY_ORDER:
                _, global_feature = self.signal.clip_vision_encoder(
                    images[name], cam_label=camera_ids, view_label=None
                )
                globals_by_modal.append(global_feature)
        finally:
            for handle in handles:
                handle.remove()
        assert all(len(captured[index]) == 3 for index in self.layers)
        stages = torch.stack(
            [torch.stack(captured[index], dim=2) for index in self.layers], dim=0
        )
        # depth, role, batch, modality, token, channel.
        baseline = torch.cat(globals_by_modal, dim=1)
        return stages, baseline


class CorrespondenceRoles(nn.Module):
    """M1 depth-conditioned evidence, M2 shared anchors, and three role operators."""

    def __init__(
        self,
        *,
        grid: tuple[int, int],
        width: int = 128,
        anchor_side: int = 4,
        m1: bool = True,
        m2: bool = True,
        mamba_factory: Callable[[int], nn.Module],
    ):
        super().__init__()
        self.grid = grid
        self.width = width
        self.anchor_side = anchor_side
        self.anchor_count = anchor_side * anchor_side
        self.m1 = m1
        self.m2 = m2
        self.depth_logits = nn.Parameter(torch.zeros(3, 3))
        self.depth_norms = nn.ModuleList(nn.LayerNorm(768) for _ in range(3))
        self.role_adapters = nn.ModuleList(
            nn.Sequential(nn.Linear(768, width), nn.GELU(), nn.Linear(width, width))
            for _ in range(3)
        )
        self.modality_embedding = nn.Parameter(torch.zeros(3, width))
        points = torch.linspace(-0.75, 0.75, anchor_side)
        yy, xx = torch.meshgrid(points, points, indexing="ij")
        self.register_buffer("reference_points", torch.stack((xx, yy), dim=-1).reshape(-1, 2))
        self.shared_offset = nn.Linear(width, 2 * self.anchor_count)
        self.modal_offset = nn.Linear(width, 2 * self.anchor_count)
        nn.init.zeros_(self.shared_offset.weight)
        nn.init.zeros_(self.shared_offset.bias)
        nn.init.zeros_(self.modal_offset.weight)
        nn.init.zeros_(self.modal_offset.bias)
        self.anchor_queries = nn.Parameter(torch.randn(self.anchor_count, width) * 0.02)
        self.cnn = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        self.transformer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=2 * width,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.mamba_norm = nn.LayerNorm(width)
        self.mamba = mamba_factory(width)
        self.cnn_to_transformer = nn.Linear(width, width, bias=False)
        self.transformer_to_mamba = nn.Linear(width, width, bias=False)
        self.output_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        if not m1:
            self.depth_logits.requires_grad_(False)
            for norm in self.depth_norms[:2]:
                for parameter in norm.parameters():
                    parameter.requires_grad_(False)
        if not m2:
            for module in (self.shared_offset, self.modal_offset,
                           self.cnn_to_transformer, self.transformer_to_mamba):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def _depth_role(self, stages: torch.Tensor, role: int) -> torch.Tensor:
        # stages: 3 depths, 3 roles, B, 3 modalities, CLS+patches, 768.
        role_stages = stages[:, role]
        if self.m1:
            weights = self.depth_logits[role].softmax(dim=0)
            combined = sum(weights[index] * self.depth_norms[index](role_stages[index])
                           for index in range(3))
        else:
            combined = self.depth_norms[-1](role_stages[-1])
        return self.role_adapters[role](combined) + self.modality_embedding[None, :, None]

    def _positions(self, pooled: torch.Tensor) -> torch.Tensor:
        batch_size = pooled.shape[0]
        base = self.reference_points[None, None].expand(batch_size, 3, -1, -1)
        if not self.m2:
            return base
        shared = self.shared_offset(pooled.mean(dim=1)).reshape(batch_size, 1, -1, 2)
        modal = self.modal_offset(pooled).reshape(batch_size, 3, -1, 2)
        return base + 0.05 * torch.tanh(shared) + 0.05 * torch.tanh(modal)

    def _sample(self, patches: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch_size, modality_count, token_count, width = patches.shape
        height, grid_width = self.grid
        assert token_count == height * grid_width and modality_count == 3
        grid = patches.reshape(batch_size * 3, height, grid_width, width).permute(0, 3, 1, 2)
        points = positions.reshape(batch_size * 3, self.anchor_count, 1, 2)
        sampled = F.grid_sample(grid, points, align_corners=True)
        return sampled.squeeze(-1).transpose(1, 2).reshape(batch_size, 3, self.anchor_count, width)

    def forward(self, stages: torch.Tensor) -> RoleEvidence:
        assert stages.ndim == 6 and stages.shape[:2] == (3, 3) and stages.shape[3] == 3
        cnn_input = self._depth_role(stages, 0)
        transformer_input = self._depth_role(stages, 1)
        mamba_input = self._depth_role(stages, 2)
        positions = self._positions(transformer_input[:, :, 0])
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
        # The sequence groups corresponding RGB/NIR/TIR anchors together.
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


class CorrespondenceTriFusion(nn.Module):
    """One deployment embedding; masked role prediction is training-only."""

    def __init__(
        self,
        signal: nn.Module,
        *,
        num_classes: int,
        grid: tuple[int, int],
        width: int = 128,
        m1: bool = True,
        m2: bool = True,
        m3: bool = True,
        mamba_factory: Callable[[int], nn.Module],
    ):
        super().__init__()
        self.backbone = CrossLayerAdaptedCLIP(signal, enabled=m1)
        self.teacher_adapters = deepcopy(self.backbone.adapters) if m3 else None
        if self.teacher_adapters is not None:
            for parameter in self.teacher_adapters.parameters():
                parameter.requires_grad_(False)
        self.roles = CorrespondenceRoles(grid=grid, width=width, m1=m1, m2=m2,
                                         mamba_factory=mamba_factory)
        self.m3 = m3
        self.teacher = deepcopy(self.roles) if m3 else None
        if self.teacher is not None:
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
        self.readout = nn.Linear(3 * width, 1536, bias=False)
        nn.init.normal_(self.readout.weight, std=0.001)
        self.readout_gain = nn.Parameter(torch.tensor(0.1))
        self.neck = nn.BatchNorm1d(1536)
        self.neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(1536, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.signal.eval()
        if self.teacher is not None:
            self.teacher.eval()
            self.teacher_adapters.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996) -> None:
        if self.teacher is not None:
            for target, source in zip(self.teacher_adapters.parameters(),
                                      self.backbone.adapters.parameters(), strict=True):
                target.lerp_(source, 1.0 - momentum)
            for target, source in zip(self.teacher.parameters(), self.roles.parameters(), strict=True):
                target.lerp_(source, 1.0 - momentum)

    def _masked_images(self, images: dict[str, torch.Tensor]):
        batch_size = images[MODALITY_ORDER[0]].shape[0]
        height, width = self.roles.grid
        missing = torch.randint(0, 3, (batch_size,), device=images[MODALITY_ORDER[0]].device)
        patch_mask = torch.rand(batch_size, 3, height, width, device=missing.device) < 0.25
        local_mask = patch_mask.clone()
        local_mask[torch.arange(batch_size, device=missing.device), missing] = False
        patch_mask[torch.arange(batch_size, device=missing.device), missing] = True
        masked = {}
        for index, name in enumerate(MODALITY_ORDER):
            image_mask = F.interpolate(patch_mask[:, index:index + 1].float(),
                                       size=images[name].shape[-2:], mode="nearest")
            masked[name] = images[name] * (1.0 - image_mask)
        return masked, local_mask, missing

    def _prediction_losses(self, images: dict[str, torch.Tensor],
                           camera_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.teacher is not None
        with torch.no_grad():
            clean_teacher, _ = self.backbone(images, camera_ids,
                                             adapter_bank=self.teacher_adapters)
            target = self.teacher(clean_teacher)
        masked_images, local_mask, missing = self._masked_images(images)
        masked_stages, _ = self.backbone(masked_images, camera_ids)
        predicted = self.roles(masked_stages)
        batch_size = clean_teacher.shape[2]
        mask_at_anchor = F.grid_sample(
            local_mask.float().reshape(batch_size * 3, 1, *self.roles.grid),
            predicted.positions.reshape(batch_size * 3, self.roles.anchor_count, 1, 2),
            mode="nearest", align_corners=True,
        ).reshape(batch_size, 3, self.roles.anchor_count)
        local_error = (predicted.cnn.float() - target.cnn.float()).square().mean(dim=-1)
        local = (local_error * mask_at_anchor).sum() / mask_at_anchor.sum().clamp_min(1)
        target_rel = F.normalize(target.transformer.float(), dim=-1)
        predicted_rel = F.normalize(predicted.transformer.float(), dim=-1)
        relation = F.mse_loss(
            predicted_rel @ predicted_rel.transpose(-1, -2),
            target_rel @ target_rel.transpose(-1, -2),
        )
        rows = torch.arange(batch_size, device=missing.device)
        cross_modal = F.mse_loss(predicted.mamba[rows, missing].float(),
                                 target.mamba[rows, missing].float())
        return {"local": local, "relation": relation, "cross_modal": cross_modal}

    def forward(self, batch: dict, *, return_aux: bool = False):
        stages, shared_global = self.backbone(batch["images"], batch["camera_ids"])
        evidence = self.roles(stages)
        pooled = torch.cat(
            [evidence.cnn.mean(dim=(1, 2)),
             evidence.transformer.mean(dim=(1, 2)),
             evidence.mamba.mean(dim=(1, 2))], dim=1
        )
        fused = F.normalize(shared_global.float() + self.readout_gain * self.readout(pooled).float(), dim=1)
        if not return_aux:
            return fused
        result = {"fused": fused, "shared_global": shared_global,
                  "logits": self.classifier(self.neck(fused))}
        if self.training and self.m3:
            result["prediction"] = self._prediction_losses(batch["images"], batch["camera_ids"])
        return result
