"""Training-only global-relation sensitivity exclusion from role reading.

Sensitivity is a model-derived proxy. Spatial exclusion does not establish
semantic non-redundancy: global context and contextualized patches remain.
"""
import torch
import torch.nn.functional as F

from .global_relation_inputs_diagnostic import GlobalInputTap, supported_global_target
from .role_global_tokens import GlobalTokenRoles
from .role_input_detach import DetachedSemanticTriFusion


class ExcludedPatchRoles(GlobalTokenRoles):
    excluded = None

    def _depth_role(self, stages, role):
        features = super()._depth_role(stages, role)
        if self.excluded is None:
            return features
        # Remove dynamic content before CNN mixing, not just after convolution.
        patches = features[:, :, 1:].masked_fill(self.excluded[..., None], 0)
        return torch.cat((features[:, :, :1], patches), dim=2)

    def sample_context(self, patches, positions, context_queries, role):
        if self.excluded is None:
            return super().sample_context(patches, positions, context_queries, role)
        with torch.autocast(patches.device.type, enabled=False):
            patches, context_queries = patches.float(), context_queries.float()
            width = patches.shape[-1]
            queries = self.anchor_queries[None] + context_queries[:, role, None]
            queries = self.query_projections[role](F.layer_norm(queries, (width,)))
            keys = self.key_projections[role](F.layer_norm(patches, (width,)))
            values = self.value_projections[role](patches)
            scores = torch.matmul(queries[:, None], keys.transpose(-1, -2)) * width ** -0.5
            # Excluded positions leave softmax; zero values alone are insufficient.
            weights = scores.masked_fill(self.excluded[:, :, None], -torch.inf).softmax(-1)
            return self.output_projections[role](torch.matmul(weights, values))


class GlobalConditionedEvidence(DetachedSemanticTriFusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            roles = ExcludedPatchRoles(token_mode='static', grid=self.roles.grid,
                width=self.roles.width, anchor_side=self.roles.anchor_side,
                m1=True, m2=True, mamba_factory=kwargs['mamba_factory'])
        roles.load_state_dict(self.roles.state_dict(), strict=True)
        self.roles = roles
        self.exposure = None
        self.mask_generator = None
        self.capture_ownership = False

    def forward_features(self, batch):
        if self.training:
            assert self.exposure in ('gradient', 'random')
            with GlobalInputTap(self.backbone) as tap:
                stages, shared_global = self.backbone(batch['images'], batch['camera_ids'])
            target, support = supported_global_target(shared_global,
                batch['identity_labels'], batch['camera_ids'])
            # Earliest retained input; no radial activation dot from the final LN.
            states = tap.ordered_states()[:3]
            gradients = torch.autograd.grad(target * 256.0, states, retain_graph=True)
            assert all(bool(torch.isfinite(g).all()) for g in gradients)
            norms = torch.stack([g[1:].detach().float().norm(dim=-1).T / 256.0
                                 for g in gradients], dim=1).to(shared_global.device)
            assert norms.shape == (shared_global.shape[0], 3, 128)
            active = norms.sum(dim=-1) > 0
            # Zero support/zero sensitivity has no exclusion, not arbitrary top-k.
            rank = norms if self.exposure == 'gradient' else torch.rand(
                norms.shape, device=norms.device, generator=self.mask_generator)
            excluded = torch.zeros_like(norms, dtype=torch.bool)
            excluded.scatter_(-1, rank.topk(32, dim=-1).indices, True)
            excluded &= active[..., None]
            self.roles.excluded = excluded
            stats = {'source': 'block3_input_global_correct_legal_margin_gradient_norm',
                'vjp_scale': 256.0, 'support': support,
                'active_sample_modalities': int(active.sum()),
                'excluded_patches': int(excluded.sum()),
                'available_sample_modalities': active.numel(),
                'gradient_norm_min': float(norms.min()),
                'gradient_norm_max': float(norms.max()),
                'gradient_norm_mean': float(norms.mean()),
                'target': float(target.detach())}
            if self.capture_ownership:
                absent = all(p.grad is None for p in self.parameters())
                assert absent
                stats['parameter_gradients_absent_after_exposure_vjp'] = absent
        else:
            stages, shared_global = self.backbone(batch['images'], batch['camera_ids'])
            stats = None
        context = F.normalize(shared_global.float().reshape(-1, 3, 512).mean(1), dim=1)
        evidence = self.role_evidence(batch, stages, context, shared_global)
        self.roles.excluded = None
        correction = self.read_evidence(evidence)
        raw = shared_global.float() + self.readout_gain * correction.float()
        return {'raw_fused': raw, 'fused': F.normalize(raw, dim=1),
                'shared_global': shared_global, 'correction': correction,
                'evidence_exposure': stats}
