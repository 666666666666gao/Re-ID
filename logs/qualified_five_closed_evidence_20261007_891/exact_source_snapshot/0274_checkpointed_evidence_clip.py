"""Recompute each modality's visual path without changing its adapter hooks."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .correspondence_roles import CrossLayerAdaptedCLIP
from .state import MODALITY_ORDER


class CheckpointedEvidenceCLIP(CrossLayerAdaptedCLIP):
    """Keep parameter names and complete batches; trade backward time for memory."""
    def __init__(self, source):
        nn.Module.__init__(self)
        assert source.enabled
        self.signal = source.signal
        self.layers = source.layers
        self.enabled = source.enabled
        self.adapters = source.adapters

    def encode_modality(self, image, camera_ids):
        captured = {index: [] for index in self.layers}
        handles = []
        for stage, index in enumerate(self.layers):
            def capture(_module, _inputs, output, *, layer=index, depth=stage):
                deltas = [adapter(output) for adapter in self.adapters[depth]]
                captured[layer].append(torch.stack(
                    [(output + delta).permute(1, 0, 2) for delta in deltas], dim=0))
                return output + sum(deltas) / 3
            handles.append(self.signal.clip_vision_encoder.base.transformer.resblocks[index]
                           .register_forward_hook(capture))
        try:
            _, feature = self.signal.clip_vision_encoder(
                image, cam_label=camera_ids, view_label=None)
        finally:
            for handle in handles:
                handle.remove()
        assert all(len(captured[index]) == 1 for index in self.layers)
        return torch.stack([captured[index][0] for index in self.layers]), feature

    def forward(self, images, camera_ids, adapter_bank=None):
        assert adapter_bank is None
        stages, globals_by_modal = [], []
        for name in MODALITY_ORDER:
            if self.training and torch.is_grad_enabled():
                stage, feature = checkpoint(self.encode_modality, images[name], camera_ids,
                                            use_reentrant=False, preserve_rng_state=True)
            else:
                stage, feature = self.encode_modality(images[name], camera_ids)
            stages.append(stage)
            globals_by_modal.append(feature)
        return torch.stack(stages, dim=3), torch.cat(globals_by_modal, dim=1)
