"""Expose actual final readout inputs only during an auxiliary training call."""
import torch

from .global_task_role_heads import GlobalTaskRoleHeads


class RegionUseHeads(GlobalTaskRoleHeads):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capture_regions = False
        self._regions = None
        self.evidence_model.roles.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, evidence):
        if self._capture_regions:
            assert self._regions is None
            self._regions = torch.stack((evidence.cnn, evidence.transformer, evidence.mamba), dim=1)

    def forward(self, batch, *, return_aux=False):
        if not return_aux:
            return super().forward(batch, return_aux=False)
        self._capture_regions = True
        output = super().forward(batch, return_aux=True)
        self._capture_regions = False
        assert self._regions is not None
        regions = self._regions
        self._regions = None
        return {**output, 'actual_regions': regions,
                'retrieval_environments': batch['retrieval_environments']}
