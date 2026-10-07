"""Match attention parameters while changing only the accessible Patch memory."""

import torch
from torch import nn
import torch.nn.functional as F

from .correspondence_context_identity import ContextSelectionRoles, ContextIdentityTriFusion


class PatchMemoryRoles(ContextSelectionRoles):
    def __init__(self, *, memory_mode, **kwargs):
        assert memory_mode in ("local", "full")
        super().__init__(**kwargs)
        self.memory_mode = memory_mode
        # Both endpoints use fixed slot addresses. No untrained offset parameters.
        for module in (self.shared_offset, self.modal_offset):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.query_projections = nn.ModuleList(nn.Linear(self.width, self.width, bias=False) for _ in range(3))
        self.key_projections = nn.ModuleList(nn.Linear(self.width, self.width, bias=False) for _ in range(3))
        self.value_projections = nn.ModuleList(nn.Linear(self.width, self.width, bias=False) for _ in range(3))
        self.output_projections = nn.ModuleList(nn.Linear(self.width, self.width, bias=False) for _ in range(3))
        height, grid_width = self.grid
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, grid_width), indexing="ij")
        patch_positions = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        # Distances are measured in Patch intervals, independent of image aspect ratio.
        intervals = torch.tensor([(grid_width - 1) / 2, (height - 1) / 2])
        distances = ((self.reference_points[:, None] - patch_positions[None]) * intervals).square().sum(-1)
        nearest = distances.argsort(dim=-1, stable=True)[:, :9]
        support = torch.zeros(self.anchor_count, height * grid_width, dtype=torch.bool)
        support.scatter_(1, nearest, True)
        # The same buffer is saved in both modes, keeping paired initial state identical.
        self.register_buffer("local_support", support)

    def _positions(self, pooled):
        return self.reference_points[None, None].expand(pooled.shape[0], 3, -1, -1)

    def sample_context(self, patches, positions, context_queries, role):
        batch, modalities, tokens, width = patches.shape
        assert modalities == 3 and tokens == self.grid[0] * self.grid[1]
        queries = self.anchor_queries[None] + context_queries[:, role, None]
        queries = self.query_projections[role](F.layer_norm(queries, (width,)))
        keys = self.key_projections[role](F.layer_norm(patches, (width,)))
        values = self.value_projections[role](patches)
        scores = torch.matmul(queries[:, None], keys.transpose(-1, -2)) * width ** -0.5
        if self.memory_mode == "local":
            scores = scores.masked_fill(~self.local_support[None, None], float("-inf"))
        selected = torch.matmul(scores.softmax(dim=-1), values)
        return self.output_projections[role](selected)


class PatchMemoryTriFusion(ContextIdentityTriFusion):
    def __init__(self, *args, memory_mode, **kwargs):
        assert kwargs["query_mode"] == "context" and kwargs["auxiliary_target"] == "none"
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            roles = PatchMemoryRoles(
                memory_mode=memory_mode, grid=self.roles.grid, width=self.roles.width,
                anchor_side=self.roles.anchor_side, m1=True, m2=True,
                mamba_factory=kwargs["mamba_factory"],
            )
        state = roles.state_dict()
        state.update(self.roles.state_dict())
        roles.load_state_dict(state, strict=True)
        self.roles = roles
        self.memory_mode = memory_mode
