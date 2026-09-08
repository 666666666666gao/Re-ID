"""MSVR310 8x16 counterpart of the sealed RGBNT201 V27 style interface.

Same MixStyle formula, modality-coupled plan and matched anchor/reference.
The original 16x8 implementation remains unchanged.
"""
from dataclasses import replace
import torch
from .signal_preserving_v8 import HierarchicalFrozenSignalBackbone
from .source_style_v27 import mix_patch_statistics
from .state import MODALITY_ORDER


class SourceStyleMSVRBackbone(HierarchicalFrozenSignalBackbone):
    """Keep the original Signal field, re-encode only the role field during training."""

    def __init__(self, signal, *, fold):
        super().__init__(signal, feature_width=512, branch_after_block=8)
        self.fold = int(fold)
        self.style_enabled = False
        self.style_plan = None
        self.last_style_stats = {}

    def forward(self, batch):
        original = super().forward(batch)
        if not self.training or self.style_plan is None:
            self.last_style_stats = {"style_active": 0, "additional_visual_passes": 0}
            return original

        plan = self.style_plan
        active = self.style_enabled and plan["active"]
        anchors, references, statistics = [], [], []

        def capture_anchor(_module, _inputs, output):
            anchors.append(output.detach())

        def capture_reference(_module, _inputs, output):
            references.append(output.detach())

        def perturb_stem(_module, _inputs, output):
            assert tuple(output.shape[1:]) == (768, 8, 16)
            mixed, stats = mix_patch_statistics(output, plan)
            statistics.append(stats)
            return mixed if active else output

        with self._split_block.register_forward_hook(capture_anchor), \
             self._reference_block.register_forward_hook(capture_reference), \
             self.signal.clip_vision_encoder.base.conv1.register_forward_hook(perturb_stem), \
             torch.no_grad():
            for modality in MODALITY_ORDER:
                self.signal.clip_vision_encoder(
                    batch["images"][modality], cam_label=batch["camera_ids"], view_label=None
                )
        assert len(anchors) == len(references) == len(statistics) == 3
        anchor = torch.stack([x.permute(1, 0, 2) for x in anchors], dim=1)
        reference = torch.stack([x.permute(1, 0, 2) for x in references], dim=1)
        anchor_change = float((anchor.float() - original.anchor_sequence.float()).abs().max())
        reference_change = float((reference.float() - original.reference_sequence.float()).abs().max())
        if active:
            assert anchor_change > 0 and reference_change > 0
        else:
            assert torch.equal(anchor, original.anchor_sequence)
            assert torch.equal(reference, original.reference_sequence)
        self.last_style_stats = {
            "style_active": int(active), "style_plan_active": int(plan["active"]),
            "additional_visual_passes": 3, "style_anchor_max_change": anchor_change,
            "style_reference_max_change": reference_change,
            **{f"style_{modality}_{key}": value
               for modality, row in zip(MODALITY_ORDER, statistics, strict=True)
               for key, value in row.items()},
        }
        return replace(original, anchor_sequence=anchor, reference_sequence=reference)
