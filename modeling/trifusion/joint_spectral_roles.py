"""Use the same semantic reading as reference for cross-spectral changes."""
import torch
import torch.nn.functional as F

from .role_input_detach import DetachedSemanticTriFusion
from .joint_spectral_delta import JointSpectralDelta, semantic_reference


class JointSpectralSemanticTriFusion(DetachedSemanticTriFusion):
    aggregation_mode = 'joint_delta'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Do not shift the existing shared/head initializer or sampler RNG.
        with torch.random.fork_rng(devices=[]):
            self.spectral_readout = JointSpectralDelta(mode=self.aggregation_mode)

    def forward_features(self, batch):
        stages, global_feature = self.backbone(batch['images'], batch['camera_ids'])
        context = F.normalize(global_feature.float().reshape(-1, 3, 512).mean(1), dim=1)
        evidence = self.role_evidence(batch, stages, context, global_feature)
        reference = semantic_reference(self.roles, stages, context)
        correction, diagnostics = self.spectral_readout(
            (evidence.cnn, evidence.transformer, evidence.mamba), reference,
            self.readout.weight)
        raw = global_feature.float() + self.readout_gain * correction
        return dict(raw_fused=raw, fused=F.normalize(raw, dim=1),
                    shared_global=global_feature, correction=correction,
                    spectral_diagnostics=diagnostics)
