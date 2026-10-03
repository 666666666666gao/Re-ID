"""A fixed tangent-update bound for V28's unit role/modality readouts.

Relative-norm control has prior art in MAG (ACL 2020); normalized directional
updates have prior art in nGPT. This is a new project hypothesis, not a novelty
claim or an identity-ranking guarantee. Original V28 code remains sealed.
"""
from dataclasses import replace
import math

import torch
from torch.nn import functional as F

from .joint_tokens_v28 import SignalPreservingV28
from .signal_preserving_v8 import SignalPreservingExpertFormationV8
from .state import EXPERT_ORDER

TANGENT_BOUND = 0.5
COSINE_FLOOR = 1 / math.sqrt(1 + TANGENT_BOUND ** 2)


def bounded_slots(h, correction):
    """Project tangentially, smoothly bound relative length, then normalize.

    Actual h norm is used so a zero correction follows the exact old bank
    normalization, without first renormalizing h a second time.
    """
    with torch.autocast(device_type=h.device.type, enabled=False):
        h = h.float()
        correction = correction.float()
        h2 = h.square().sum(dim=-1, keepdim=True)
        tangent = correction - (correction * h).sum(dim=-1, keepdim=True) / h2 * h
        update = tangent * torch.rsqrt(1 + tangent.square().sum(dim=-1, keepdim=True)
                                      / (TANGENT_BOUND ** 2 * h2))
        return F.normalize(h + update, dim=-1), update


def bounded_joint_bank(modal, corrections):
    roles, ratios, cosines, orthogonality = [], [], [], []
    for name in EXPERT_ORDER:
        h = modal[name]
        slots, update = bounded_slots(h, corrections[name])
        roles.append(F.normalize(slots.flatten(1), dim=1))
        with torch.no_grad():
            h = h.float()
            h2 = h.square().sum(dim=-1)
            ratios.append((update.detach().square().sum(dim=-1) / h2).sqrt())
            cosines.append((slots.detach() * h).sum(dim=-1) / h2.sqrt())
            orthogonality.append((update.detach() * h).sum(dim=-1).abs() / h2)
    stats = {
        'geometry_bound': TANGENT_BOUND,
        'geometry_update_ratio_max': float(torch.stack(ratios).max()),
        'geometry_update_ratio_mean': float(torch.stack(ratios).mean()),
        'geometry_cosine_min': float(torch.stack(cosines).min()),
        'geometry_cosine_mean': float(torch.stack(cosines).mean()),
        'geometry_tangent_relative_dot_max': float(torch.stack(orthogonality).max()),
        'geometry_slot_observations': int(sum(v.numel() for v in ratios)),
    }
    assert stats['geometry_update_ratio_max'] <= TANGENT_BOUND + 2e-6
    assert stats['geometry_cosine_min'] >= COSINE_FLOOR - 2e-6
    return F.normalize(torch.cat(roles, dim=1), dim=1), stats


class SignalPreservingV29(SignalPreservingV28):
    """Keep V28's collection/supervision paths; bound the fused slot update."""

    def forward(self, batch, targets=None, return_aux=False, retrieval_output='fused'):
        captured = {}

        def field_hook(_module, _inputs, output):
            captured['field'] = output

        def cnn_hook(_module, _inputs, output):
            captured['cnn'] = output

        def transformer_hook(_module, _inputs, output):
            captured['transformer'] = output[:, 1:]

        def mamba_hook(_module, _inputs, output):
            captured['mamba'] = output

        def fusion_hook(_module, inputs, output):
            baseline, representations = inputs
            reference = captured['field'].reference_sequence[:, :, 1:]
            deltas = torch.stack([
                captured[name].reshape_as(reference) - reference for name in EXPERT_ORDER
            ], dim=1)
            modal = representations.modal_residual_embeddings
            corrections = self.joint(deltas) if self.joint_enabled else {
                name: torch.zeros_like(modal[name]) for name in EXPERT_ORDER
            }
            bank, geometry = bounded_joint_bank(modal, corrections)
            fused = torch.cat((baseline, bank * baseline.detach().norm(dim=1, keepdim=True)), dim=1)
            self.last_joint_stats = {
                'joint_enabled': int(self.joint_enabled), 'joint_sequence_tokens': 1152,
                'joint_token_input_requires_grad': int(deltas.requires_grad),
                'joint_correction_abs_mean': float(torch.stack([
                    value.detach().float().abs().mean() for value in corrections.values()
                ]).mean()),
                'joint_fused_change_abs_max': float((fused.detach() - output.fused_embedding.detach()).abs().max()),
                **geometry,
            }
            return replace(output, fused_embedding=fused)

        with self.baseline.register_forward_hook(field_hook), \
             self.encoder.cnn_stages[-1].register_forward_hook(cnn_hook), \
             self.encoder.transformer_stages[-1].register_forward_hook(transformer_hook), \
             self.encoder.mamba_stages[-1].register_forward_hook(mamba_hook), \
             self.fusion.register_forward_hook(fusion_hook):
            return SignalPreservingExpertFormationV8.forward(
                self, batch, targets, return_aux, retrieval_output)
