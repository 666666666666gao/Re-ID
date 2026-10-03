"""Place the original full-batch visual graph on two CUDA devices."""
from types import MethodType

import torch

from .state import MODALITY_ORDER


def partition_visual_graph(backbone):
    """One parameter bank, original arithmetic, first6/last6 placement."""
    assert torch.cuda.device_count() == 2
    assert backbone.enabled and backbone.layers == (3, 7, 11)
    encoder = backbone.signal.clip_vision_encoder
    visual = encoder.base
    assert not visual.prompt_sign and not visual.adapter_sign
    assert len(visual.transformer.resblocks) == 12
    original_parameters = {name: id(value) for name, value in backbone.named_parameters()}
    assert all(value.grad is None for value in backbone.parameters())
    for parameter in (visual.class_embedding, visual.positional_embedding, encoder.cv_embed):
        parameter.data = parameter.data.to('cuda:1')
    visual.conv1.to('cuda:1')
    visual.ln_pre.to('cuda:1')
    for block in visual.transformer.resblocks[:6]:
        block.to('cuda:1')
    backbone.adapters[0].to('cuda:1')
    assert original_parameters == {name: id(value) for name, value in backbone.named_parameters()}

    # These two fixed transfers do not alter the author's block implementation.
    encoder.register_forward_pre_hook(
        lambda _module, inputs: (inputs[0].to('cuda:1'), *inputs[1:]))
    visual.transformer.resblocks[6].register_forward_pre_hook(
        lambda _module, inputs: (inputs[0].to('cuda:0'), *inputs[1:]))
    backbone.forward = MethodType(_partitioned_forward, backbone)


def _partitioned_forward(self, images, camera_ids, adapter_bank=None):
    assert adapter_bank is None
    captured = {index: [] for index in self.layers}
    handles = []
    for stage_index, index in enumerate(self.layers):
        def capture(_module, _inputs, output, *, layer=index, stage=stage_index):
            deltas = [adapter(output) for adapter in self.adapters[stage]]
            # Same original role stack; transfer it to the role/head device.
            snapshot = torch.stack(
                [(output + delta).permute(1, 0, 2) for delta in deltas], dim=0)
            captured[layer].append(snapshot.to('cuda:0'))
            return output + sum(deltas) / 3
        handles.append(self.signal.clip_vision_encoder.base.transformer.resblocks[index].register_forward_hook(capture))
    globals_by_modal = []
    try:
        for name in MODALITY_ORDER:
            _, global_feature = self.signal.clip_vision_encoder(
                images[name], cam_label=camera_ids, view_label=None)
            globals_by_modal.append(global_feature)
    finally:
        for handle in handles:
            handle.remove()
    assert all(len(captured[index]) == 3 for index in self.layers)
    stages = torch.stack(
        [torch.stack(captured[index], dim=2) for index in self.layers], dim=0)
    return stages, torch.cat(globals_by_modal, dim=1)


def placement_summary(model):
    groups = {}
    for name, parameter in model.named_parameters():
        device = str(parameter.device)
        row = groups.setdefault(device, {'parameters':0, 'trainable_parameters':0,
            'parameter_bytes':0, 'gradient_adam_bytes_if_fp32':0})
        row['parameters'] += parameter.numel()
        row['parameter_bytes'] += parameter.numel() * parameter.element_size()
        if parameter.requires_grad:
            row['trainable_parameters'] += parameter.numel()
            row['gradient_adam_bytes_if_fp32'] += parameter.numel() * 12
    assert set(groups) == {'cuda:0', 'cuda:1'}
    return groups
