"""Process layer evidence before aggregation, with optional persistent role state."""

import torch
from .correspondence_context_identity import ContextIdentityTriFusion, ContextSelectionRoles
from .correspondence_roles import RoleEvidence


class CrossDepthStateRoles(ContextSelectionRoles):
    def __init__(self, *, depth_mode, **kwargs):
        super().__init__(**kwargs)
        assert depth_mode in ('mixed_once', 'depth_mean', 'depth_recurrent')
        self.depth_mode = depth_mode
        # All three controls use the same uniform address predictor and parameter set.
        self.depth_logits.requires_grad_(False)

    def _at_depth(self, stages, depth, role):
        value = self.depth_norms[depth](stages[depth, role])
        return self.role_adapters[role](value) + self.modality_embedding[None, :, None]

    def _step(self, inputs, positions, context_queries, previous=None):
        cnn_input, transformer_input, mamba_input = inputs
        height, grid_width = self.grid
        cnn_grid = cnn_input[:, :, 1:].reshape(-1, height, grid_width, self.width).permute(0, 3, 1, 2)
        cnn_grid = cnn_grid + self.cnn(cnn_grid)
        cnn_patches = cnn_grid.permute(0, 2, 3, 1).reshape(cnn_input.shape[0], 3, -1, self.width)
        cnn = self.sample_context(cnn_patches, positions, context_queries, 0)
        transformer = self.sample_context(transformer_input[:, :, 1:], positions, context_queries, 1)
        mamba = self.sample_context(mamba_input[:, :, 1:], positions, context_queries, 2)
        slots = self.anchor_queries[None, None]
        cnn = cnn + slots
        if previous is not None:
            cnn = cnn + previous.cnn
        cnn = self.output_norms[0](cnn)
        transformer = transformer + slots + self.cnn_to_transformer(cnn)
        if previous is not None:
            transformer = transformer + previous.transformer
        transformer = self.transformer(transformer.flatten(0, 1)).reshape_as(transformer)
        transformer = self.output_norms[1](transformer)
        mamba = mamba + slots + self.transformer_to_mamba(transformer)
        if previous is not None:
            mamba = mamba + previous.mamba
        sequence = mamba.permute(0, 2, 1, 3).reshape(-1, self.anchor_count * 3, self.width)
        sequence = self.mamba_norm(sequence)
        propagated = .5 * (self.mamba(sequence) + self.mamba(sequence.flip(1)).flip(1))
        mamba = self.output_norms[2](
            propagated.reshape(-1, self.anchor_count, 3, self.width).permute(0, 2, 1, 3))
        return RoleEvidence(cnn=cnn, transformer=transformer, mamba=mamba, positions=positions)

    def forward(self, stages, context):
        assert stages.ndim == 6 and stages.shape[:2] == (3, 3) and stages.shape[3] == 3
        assert context.shape == (stages.shape[2], 512)
        context_queries = self.context_queries(context).reshape(context.shape[0], 3, self.width)
        mixed = [self._depth_role(stages, role) for role in range(3)]
        # Every condition uses the same addresses; only evidence processing changes.
        positions = self._positions(mixed[1][:, :, 0])
        if self.depth_mode == 'mixed_once':
            return self._step(mixed, positions, context_queries)
        outputs = []
        previous = None
        for depth in range(3):
            inputs = [self._at_depth(stages, depth, role) for role in range(3)]
            evidence = self._step(inputs, positions, context_queries, previous)
            outputs.append(evidence)
            if self.depth_mode == 'depth_recurrent':
                previous = evidence
        if self.depth_mode == 'depth_recurrent':
            return outputs[-1]
        return RoleEvidence(
            cnn=torch.stack([item.cnn for item in outputs]).mean(dim=0),
            transformer=torch.stack([item.transformer for item in outputs]).mean(dim=0),
            mamba=torch.stack([item.mamba for item in outputs]).mean(dim=0), positions=positions)


class CrossDepthRoleStateTriFusion(ContextIdentityTriFusion):
    def __init__(self, *args, depth_mode, **kwargs):
        assert kwargs['query_mode'] == 'context' and kwargs['auxiliary_target'] == 'none'
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            roles = CrossDepthStateRoles(
                depth_mode=depth_mode, grid=self.roles.grid, width=self.roles.width,
                anchor_side=self.roles.anchor_side, m1=True, m2=True,
                mamba_factory=kwargs['mamba_factory'])
        roles.load_state_dict(self.roles.state_dict(), strict=True)
        self.roles = roles
        self.depth_mode = depth_mode
