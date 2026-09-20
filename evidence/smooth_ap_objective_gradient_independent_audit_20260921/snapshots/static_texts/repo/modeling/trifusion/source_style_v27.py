"""Training-only cross-camera patch-stem statistic mixing for original V8 roles.

Independent implementation of the MixStyle moment formula (Zhou et al., ICLR 2021).
This CLIP-stem, modality-coupled use is a new hypothesis, not an author reproduction.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from .signal_preserving_v8 import HierarchicalFrozenSignalBackbone
from .state import MODALITY_ORDER


def make_style_plan(cameras, *, fold, step, force_active=False):
    cameras = np.asarray(cameras, dtype=np.int64)
    allowed = cameras[:, None] != cameras[None, :]
    assert allowed.any(axis=1).all()
    rng = np.random.default_rng(np.random.SeedSequence([42, int(fold), int(step)]))
    active = bool(rng.random() < 0.5)
    choices = rng.random(allowed.shape)
    choices[~allowed] = np.inf
    donors = choices.argmin(axis=1)
    coefficients = rng.beta(0.1, 0.1, size=len(cameras))
    assert np.all(cameras[donors] != cameras)
    return {
        "fold": int(fold), "step": int(step), "active": active or bool(force_active),
        "forced_active": bool(force_active), "donors": donors.tolist(),
        "coefficients": coefficients.tolist(), "all_donors_cross_camera": True,
    }


def mix_patch_statistics(value, plan):
    assert value.ndim == 4
    x = value.float()
    mean = x.mean(dim=(2, 3), keepdim=True).detach()
    variance = x.var(dim=(2, 3), unbiased=False, keepdim=True).detach()
    std = (variance + 1e-6).sqrt()
    donor = torch.tensor(plan["donors"], dtype=torch.long, device=x.device)
    weight = torch.tensor(plan["coefficients"], dtype=torch.float32, device=x.device)[:, None, None, None]
    mixed_mean = weight * mean + (1 - weight) * mean[donor]
    mixed_std = weight * std + (1 - weight) * std[donor]
    mixed = ((x - mean) / std * mixed_std + mixed_mean).to(value.dtype)
    assert bool(torch.isfinite(mixed).all())
    return mixed, {
        "variance_min": float(variance.min()),
        "mean_change_abs_mean": float((mixed_mean - mean).abs().mean()),
        "std_change_abs_mean": float((mixed_std - std).abs().mean()),
        "proposed_change_abs_mean": float((mixed.float() - x).abs().mean()),
    }


class SourceStyleFrozenSignalBackbone(HierarchicalFrozenSignalBackbone):
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
            assert tuple(output.shape[1:]) == (768, 16, 8)
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
