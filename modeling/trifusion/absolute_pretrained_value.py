"""Retain the image-common component of the role-selected pretrained values."""
import torch
import torch.nn.functional as F

from .pretrained_value_delta import PretrainedValueDeltaTriFusion, value_delta_readout
from .joint_spectral_delta import semantic_reference
from .native_clip_patch_evidence import project_attention_patches


def absolute_value_readout(roles, processed, reference, keys_by_role, values):
    _, detail = value_delta_readout(
        roles, processed, reference, keys_by_role, values, 'uniform_reference')
    image_mean = values.float().mean(2)
    regions = detail['regions'] + image_mean[:, None, :, None, :]
    correction = regions.mean(dim=(1, 3)).flatten(1)
    return correction, dict(detail, regions=regions, value_image_mean=image_mean,
                            attention_weights=detail['attention_change'] + 1. / 128)


class AbsolutePretrainedValueTriFusion(PretrainedValueDeltaTriFusion):
    """Same parameters and tasks as the qualified differential-value control."""
    reference_mode = 'uniform_reference'

    def forward_features(self, batch):
        captured = []
        visual = self.backbone.signal.clip_vision_encoder.base

        def capture(_module, _inputs, output):
            with torch.no_grad(), torch.autocast(output[0].device.type, enabled=False):
                captured.append(project_attention_patches(visual, output[0].detach()).detach())

        handle = visual.transformer.resblocks[-1].attn.register_forward_hook(capture)
        stages, global_feature = self.backbone(batch['images'], batch['camera_ids'])
        handle.remove()
        assert len(captured) == 3
        values = torch.stack(captured, dim=1)
        context = F.normalize(global_feature.float().reshape(-1, 3, 512).mean(1), dim=1)
        evidence = self.role_evidence(batch, stages, context, global_feature)
        reference = semantic_reference(self.roles, stages, context)
        keys = [self.roles._depth_role(stages.detach(), role)[:, :, 1:] for role in range(3)]
        correction, diagnostics = absolute_value_readout(self.roles,
            (evidence.cnn, evidence.transformer, evidence.mamba), reference, keys, values)
        raw = global_feature.float() + self.readout_gain * correction
        return dict(raw_fused=raw, fused=F.normalize(raw, dim=1),
                    shared_global=global_feature, correction=correction, value_diagnostics=diagnostics)
