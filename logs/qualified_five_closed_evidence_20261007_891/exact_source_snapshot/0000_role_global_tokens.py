"""Separate identity-conditioned role interaction from direct global projection."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .correspondence_roles import RoleEvidence
from .slot_competition_fp32_roles import FP32SlotCompetitionRoles, FP32SlotCompetitionTriFusion


TOKEN_MODES = ("static", "token", "direct")


class GlobalTokenRoles(FP32SlotCompetitionRoles):
    def __init__(self, *, token_mode, **kwargs):
        assert token_mode in TOKEN_MODES
        super().__init__(attention_normalization="independent", **kwargs)
        self.token_mode = token_mode
        self.global_projection = nn.Linear(512, self.width, bias=False)
        nn.init.zeros_(self.global_projection.weight)
        self.register_buffer("static_token_context", torch.ones(512) / math.sqrt(512))

    def projected_global(self, shared_global):
        per_modal = F.normalize(shared_global.float().reshape(-1, 3, 512), dim=-1)
        return self.global_projection(per_modal)

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
        transformer = self.sample_context(transformer_input[:, :, 1:], positions, context_queries, 1)
        mamba = self.sample_context(mamba_input[:, :, 1:], positions, context_queries, 2)
        queries = self.anchor_queries[None, None]
        cnn = self.output_norms[0](cnn + queries)
        transformer = transformer + queries + self.cnn_to_transformer(cnn)
        if self.token_mode == "token":
            global_token = self.projected_global(shared_global)
        else:
            fixed = self.static_token_context[None, None].expand(context.shape[0], 3, -1)
            global_token = self.global_projection(fixed)
        # The extra token is an interaction input; exclude it from role readout.
        sequence = torch.cat((global_token.unsqueeze(2), transformer), dim=2)
        transformer = self.transformer(sequence.flatten(0, 1))[:, 1:].reshape_as(transformer)
        transformer = self.output_norms[1](transformer)
        mamba_input = mamba + queries + self.transformer_to_mamba(transformer)
        sequence = mamba_input.permute(0, 2, 1, 3).reshape(-1, self.anchor_count * 3, self.width)
        sequence = self.mamba_norm(sequence)
        propagated = 0.5 * (self.mamba(sequence) + self.mamba(sequence.flip(1)).flip(1))
        mamba = self.output_norms[2](
            propagated.reshape(-1, self.anchor_count, 3, self.width).permute(0, 2, 1, 3)
        )
        return RoleEvidence(cnn=cnn, transformer=transformer, mamba=mamba, positions=positions)


class GlobalTokenTriFusion(FP32SlotCompetitionTriFusion):
    def __init__(self, *args, token_mode, **kwargs):
        super().__init__(*args, attention_normalization="independent", **kwargs)
        with torch.random.fork_rng(devices=[]):
            roles = GlobalTokenRoles(
                token_mode=token_mode, grid=self.roles.grid, width=self.roles.width,
                anchor_side=self.roles.anchor_side, m1=True, m2=True,
                mamba_factory=kwargs["mamba_factory"],
            )
        state = roles.state_dict()
        state.update(self.roles.state_dict())
        roles.load_state_dict(state, strict=True)
        self.roles = roles
        self.token_mode = token_mode

    def forward(self, batch, *, return_aux=False):
        stages, shared_global = self.backbone(batch["images"], batch["camera_ids"])
        context = F.normalize(shared_global.float().reshape(-1, 3, 512).mean(dim=1), dim=1)
        evidence = self.roles(stages, context, shared_global)
        correction = self.read_evidence(evidence)
        if self.token_mode == "direct":
            projected = self.roles.projected_global(shared_global)[:, :, None].expand(-1, -1, 16, -1)
            direct = RoleEvidence(cnn=projected, transformer=projected, mamba=projected,
                                  positions=evidence.positions)
            correction = correction + self.read_evidence(direct)
        fused = F.normalize(shared_global.float() + self.readout_gain * correction.float(), dim=1)
        if not return_aux:
            return fused
        return {"fused": fused, "shared_global": shared_global,
                "joint_local": F.normalize(correction.float(), dim=1),
                "logits": self.classifier(self.neck(fused))}
