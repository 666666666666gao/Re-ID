"""Independent shared-global control for the correspondence-role model."""

import torch.nn.functional as F
from torch import nn

from .correspondence_roles import CorrespondenceTriFusion


class CorrespondenceGlobalOnly(nn.Module):
    """Keep the initialized M1/global/head modules; omit the role path."""

    def __init__(self, initialized: CorrespondenceTriFusion):
        super().__init__()
        assert initialized.backbone.enabled and not initialized.m3
        assert not initialized.roles.m2
        self.backbone = initialized.backbone
        self.neck = initialized.neck
        self.classifier = initialized.classifier

    def forward(self, batch: dict, *, return_aux: bool = False):
        _, shared_global = self.backbone(batch["images"], batch["camera_ids"])
        fused = F.normalize(shared_global.float(), dim=1)
        if not return_aux:
            return fused
        return {"fused": fused, "shared_global": shared_global,
                "logits": self.classifier(self.neck(fused))}
