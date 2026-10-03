"""Keep the original CLIP graph while storing saved backward tensors on CPU."""
from types import MethodType

import torch


def cpu_saved_forward(backbone, *args, **kwargs):
    if backbone.training and torch.is_grad_enabled():
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            return backbone._cpu_saved_original_forward(*args, **kwargs)
    return backbone._cpu_saved_original_forward(*args, **kwargs)


def save_visual_tensors_on_cpu(backbone):
    backbone._cpu_saved_original_forward = backbone.forward
    backbone.forward = MethodType(cpu_saved_forward, backbone)
