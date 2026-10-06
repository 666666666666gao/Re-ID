"""Let fused-role identity heads adapt without changing the global task."""
from copy import deepcopy

from torch import nn
import torch.nn.functional as F

from .global_task_role_heads import GlobalTaskRoleHeads


class IndependentRoleHeads(GlobalTaskRoleHeads):
    def __init__(self, evidence_model):
        super().__init__(evidence_model)
        self.role_heads = nn.ModuleList([
            nn.ModuleDict({'neck': deepcopy(getattr(self.signal, neck)),
                           'classifier': deepcopy(getattr(self.signal, classifier))})
            for neck, classifier in self.head_names
        ])

    def forward(self, batch, *, return_aux=False):
        output = self.evidence_model.forward_features(batch)
        if not return_aux:
            return output['fused']
        global_feature = output['shared_global'].float()
        raw = global_feature.detach() + self.evidence_model.readout_gain * output['correction'].float()
        global_parts = (global_feature,) if self.signal.direct else global_feature.split(512, dim=1)
        fused_parts = (raw,) if self.signal.direct else raw.split(512, dim=1)
        global_heads, fused_heads = [], []
        for (neck, classifier), role_head, global_part, fused_part in zip(
                self.head_names, self.role_heads, global_parts, fused_parts):
            global_heads.append((getattr(self.signal, classifier)(
                getattr(self.signal, neck)(global_part)), global_part))
            fused_heads.append((role_head['classifier'](role_head['neck'](fused_part)), fused_part))
        return {**output, 'raw_fused': raw, 'fused': F.normalize(raw, dim=1),
                'heads': fused_heads, 'global_heads': global_heads}
