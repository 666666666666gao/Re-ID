"""Condition local selection without copying global values into role evidence."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .correspondence_evidence_readout import ContentSelectionRoles, EvidenceReadoutTriFusion
from .correspondence_roles import RoleEvidence


class ContextSelectionRoles(ContentSelectionRoles):
    def __init__(self, **kwargs):
        super().__init__(selection="query", **kwargs)
        self.context_queries = nn.Linear(512, 3 * self.width, bias=False)
        nn.init.zeros_(self.context_queries.weight)

    def sample_context(self, patches, positions, context_queries, role):
        batch, modalities, tokens, width = patches.shape
        height, grid_width = self.grid
        assert modalities == 3 and tokens == height * grid_width
        grid = patches.reshape(batch * 3, height, grid_width, width).permute(0, 3, 1, 2)
        points = positions.unsqueeze(-2) + self.candidate_offsets
        samples = F.grid_sample(
            grid, points.reshape(batch * 3, self.anchor_count * 9, 1, 2), align_corners=True,
        ).squeeze(-1).transpose(1, 2).reshape(batch, 3, self.anchor_count, 9, width)
        queries = self.anchor_queries[None] + context_queries[:, role, None]
        queries = F.layer_norm(queries, (width,))[:, None, :, None]
        keys = F.layer_norm(samples, (width,))
        scores = (queries * keys).sum(dim=-1) * width ** -0.5
        return (scores.softmax(dim=-1).unsqueeze(-1) * samples).sum(dim=-2)

    def forward(self, stages, context):
        assert stages.ndim == 6 and stages.shape[:2] == (3, 3) and stages.shape[3] == 3
        assert context.shape == (stages.shape[2], 512)
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
        transformer = self.sample_context(transformer_input[:, :, 1:], positions, context_queries, 1)
        mamba = self.sample_context(mamba_input[:, :, 1:], positions, context_queries, 2)
        # Conditioning changes candidate weights only. Slot messages contain no global values.
        queries = self.anchor_queries[None, None]
        cnn = self.output_norms[0](cnn + queries)
        transformer = transformer + queries + self.cnn_to_transformer(cnn)
        transformer = self.transformer(transformer.flatten(0, 1)).reshape_as(transformer)
        transformer = self.output_norms[1](transformer)
        mamba_input = mamba + queries + self.transformer_to_mamba(transformer)
        sequence = mamba_input.permute(0, 2, 1, 3).reshape(-1, self.anchor_count * 3, self.width)
        sequence = self.mamba_norm(sequence)
        propagated = 0.5 * (self.mamba(sequence) + self.mamba(sequence.flip(1)).flip(1))
        mamba = self.output_norms[2](
            propagated.reshape(-1, self.anchor_count, 3, self.width).permute(0, 2, 1, 3)
        )
        return RoleEvidence(cnn=cnn, transformer=transformer, mamba=mamba, positions=positions)


class ContextIdentityTriFusion(EvidenceReadoutTriFusion):
    def __init__(self, *args, query_mode, auxiliary_target, **kwargs):
        assert query_mode in ("static", "context")
        assert auxiliary_target in ("none", "local", "global")
        assert kwargs["m1"] and kwargs["m2"] and not kwargs["m3"] and kwargs["width"] == 128
        super().__init__(*args, selection="query", structured_readout=True, **kwargs)
        with torch.random.fork_rng(devices=[]):
            roles = ContextSelectionRoles(
                grid=self.roles.grid, width=self.roles.width, anchor_side=self.roles.anchor_side,
                m1=True, m2=True, mamba_factory=kwargs["mamba_factory"],
            )
            self.auxiliary_neck = nn.BatchNorm1d(1536)
            self.auxiliary_neck.bias.requires_grad_(False)
            self.auxiliary_classifier = nn.Linear(1536, kwargs["num_classes"], bias=False)
            nn.init.normal_(self.auxiliary_classifier.weight, std=0.001)
        state = roles.state_dict()
        state.update(self.roles.state_dict())
        roles.load_state_dict(state, strict=True)
        self.roles = roles
        self.query_mode = query_mode
        self.auxiliary_target = auxiliary_target
        self.register_buffer("static_context", torch.ones(512) / math.sqrt(512))
        if auxiliary_target == "none":
            for module in (self.auxiliary_neck, self.auxiliary_classifier):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def forward(self, batch, *, return_aux=False):
        stages, shared_global = self.backbone(batch["images"], batch["camera_ids"])
        if self.query_mode == "context":
            context = F.normalize(shared_global.float().reshape(-1, 3, 512).mean(dim=1), dim=1)
        else:
            context = self.static_context[None].expand(shared_global.shape[0], -1)
        evidence = self.roles(stages, context)
        correction = self.read_evidence(evidence)
        local = F.normalize(correction.float(), dim=1)
        fused = F.normalize(shared_global.float() + self.readout_gain * correction.float(), dim=1)
        if not return_aux:
            return fused
        result = {"fused": fused, "shared_global": shared_global, "joint_local": local,
                  "logits": self.classifier(self.neck(fused))}
        if self.auxiliary_target != "none":
            auxiliary = local if self.auxiliary_target == "local" else F.normalize(shared_global.float(), dim=1)
            result["auxiliary_logits"] = self.auxiliary_classifier(self.auxiliary_neck(auxiliary))
        return result
