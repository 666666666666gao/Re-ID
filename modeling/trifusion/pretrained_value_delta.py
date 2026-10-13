"""Roles select changes in the existing CLIP visual coordinate system."""
import torch
import torch.nn.functional as F
from torch import nn

from .role_input_detach import DetachedSemanticTriFusion
from .joint_spectral_delta import semantic_reference
from .native_clip_patch_evidence import project_attention_patches


def value_delta_readout(roles, processed, reference, keys_by_role, values, reference_mode):
    """Compare readings, then use the detached original 512-dimensional values.

    Both references retain zero total attention change. This cancels a common
    value offset conditional on the weights, not arbitrary image/style factors.
    """
    assert reference_mode in ('semantic_reference', 'uniform_reference')
    assert values.shape[1:] == (3, 128, 512) and not values.requires_grad
    changes = []
    with torch.autocast(values.device.type, enabled=False):
        for role in range(3):
            actual, baseline = processed[role].float(), reference[role].float()
            patches = keys_by_role[role].float()
            assert actual.shape[1:] == baseline.shape[1:] == (3, 16, 128)
            assert patches.shape[1:] == (3, 128, 128)
            keys = roles.key_projections[role](F.layer_norm(patches, (128,)))
            query = roles.query_projections[role](F.layer_norm(actual, (128,)))
            weight = (query @ keys.transpose(-1, -2) * 128 ** -.5).softmax(-1)
            if reference_mode == 'semantic_reference':
                query_ref = roles.query_projections[role](F.layer_norm(baseline, (128,)))
                weight_ref = (query_ref @ keys.transpose(-1, -2) * 128 ** -.5).softmax(-1)
            else:
                weight_ref = torch.full_like(weight, 1. / 128)
            delta = weight - weight_ref
            changes.append(delta)
        changes = torch.stack(changes, dim=1)
        regions = changes @ values[:, None].float()
        correction = regions.mean(dim=(1, 3)).flatten(1)
    assert correction.shape == (values.shape[0], 1536)
    return correction, dict(regions=regions, attention_change=changes)


class PretrainedValueDeltaTriFusion(DetachedSemanticTriFusion):
    reference_mode = 'semantic_reference'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This path uses the pretrained visual basis directly, so the old free
        # 384->1536 readout has no task and is removed rather than left idle.
        assert self.readout.weight.shape == (1536, 384)
        self.readout = nn.Identity()

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
        correction, diagnostics = value_delta_readout(self.roles,
            (evidence.cnn, evidence.transformer, evidence.mamba), reference, keys, values, self.reference_mode)
        raw = global_feature.float() + self.readout_gain * correction
        return dict(raw_fused=raw, fused=F.normalize(raw, dim=1),
            shared_global=global_feature, correction=correction, value_diagnostics=diagnostics)
