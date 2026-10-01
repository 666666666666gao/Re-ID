"""Separate private role snapshots from shared CLIP adapter writeback."""
from copy import deepcopy

import torch
from torch import nn

from .correspondence_roles import CrossLayerAdaptedCLIP
from .state import MODALITY_ORDER


class SharedPrivateEvidenceCLIP(CrossLayerAdaptedCLIP):
    """Matched private banks; the control also writes their mean to the shared stream."""

    def __init__(self, source: CrossLayerAdaptedCLIP, *, private_writeback: bool):
        nn.Module.__init__(self)
        assert source.enabled
        self.signal = source.signal
        self.layers = source.layers
        self.enabled = True
        self.adapters = source.adapters
        self.private_adapters = deepcopy(source.adapters)
        self.private_writeback = private_writeback

    def forward(self, images, camera_ids, adapter_bank=None):
        # This experiment has no M3 teacher or alternative adapter bank.
        assert adapter_bank is None
        captured = {index: [] for index in self.layers}
        handles = []
        for stage, index in enumerate(self.layers):
            def capture(_module, _inputs, output, *, layer=index, depth=stage):
                shared = output + sum(adapter(output) for adapter in self.adapters[depth]) / 3
                private = [adapter(shared) for adapter in self.private_adapters[depth]]
                captured[layer].append(torch.stack(
                    [(shared + delta).permute(1, 0, 2) for delta in private], dim=0
                ))
                return shared + sum(private) / 3 if self.private_writeback else shared
            handles.append(self.signal.clip_vision_encoder.base.transformer.resblocks[index]
                           .register_forward_hook(capture))
        globals_by_modal = []
        try:
            for name in MODALITY_ORDER:
                _, feature = self.signal.clip_vision_encoder(
                    images[name], cam_label=camera_ids, view_label=None
                )
                globals_by_modal.append(feature)
        finally:
            for handle in handles:
                handle.remove()
        assert all(len(captured[index]) == 3 for index in self.layers)
        stages = torch.stack(
            [torch.stack(captured[index], dim=2) for index in self.layers], dim=0
        )
        return stages, torch.cat(globals_by_modal, dim=1)
