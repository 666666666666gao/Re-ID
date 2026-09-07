"""Joint pre-pooling role/modality residual aggregation; original V8 heads retained.

Full-sequence aggregation is inspired by MambaPro (AAAI 2025), not an author
reproduction. This implementation uses the existing project Mamba primitive.
"""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .experts.mamba import production_mamba_factory
from .signal_preserving_v8 import SignalPreservingExpertFormationV8
from .state import EXPERT_ORDER


class JointResidualTokens(nn.Module):
    """One shared bidirectional scan over role-major/modal-major spatial tokens."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.down = nn.Linear(768, 128)
        self.mixer = production_mamba_factory(128)
        self.up = nn.ModuleDict({name: nn.Linear(128, 512, bias=False) for name in EXPERT_ORDER})
        for projection in self.up.values():
            nn.init.zeros_(projection.weight)

    def forward(self, deltas):
        assert deltas.ndim == 5 and tuple(deltas.shape[1:]) == (3, 3, 128, 768)
        batch = deltas.shape[0]
        sequence = self.down(self.norm(deltas)).reshape(batch, 1152, 128)
        mixed = (self.mixer(sequence) + self.mixer(sequence.flip(1)).flip(1)) * 0.5
        pooled = (sequence + mixed).reshape(batch, 3, 3, 128, 128).mean(dim=3)
        return {name: self.up[name](pooled[:, index]) for index, name in enumerate(EXPERT_ORDER)}


def normalized_joint_bank(modal, corrections):
    roles = []
    for name in EXPERT_ORDER:
        slots = F.normalize(modal[name] + corrections[name], dim=-1)
        roles.append(F.normalize(slots.flatten(1), dim=1))
    return F.normalize(torch.cat(roles, dim=1), dim=1)


class SignalPreservingV28(SignalPreservingExpertFormationV8):
    """Share the loaded V8 objects; change only the fused bank before its old head."""

    def __init__(self, original, *, joint_enabled):
        nn.Module.__init__(self)
        for name, module in original.named_children():
            self.add_module(name, module)
        for name in ("num_classes", "baseline_embedding_width", "fused_embedding_width",
                     "branch_embedding_width", "residual_embedding_width"):
            setattr(self, name, getattr(original, name))
        self.joint_enabled = bool(joint_enabled)
        self.joint = JointResidualTokens() if self.joint_enabled else None
        self.last_joint_stats = {}

    def forward(self, batch, targets=None, return_aux=False, retrieval_output="fused"):
        captured = {}

        def field_hook(_module, _inputs, output):
            captured["field"] = output

        def cnn_hook(_module, _inputs, output):
            captured["cnn"] = output

        def transformer_hook(_module, _inputs, output):
            captured["transformer"] = output[:, 1:]

        def mamba_hook(_module, _inputs, output):
            captured["mamba"] = output

        def fusion_hook(_module, inputs, output):
            baseline, representations = inputs
            reference = captured["field"].reference_sequence[:, :, 1:]
            deltas = torch.stack([
                captured[name].reshape_as(reference) - reference for name in EXPERT_ORDER
            ], dim=1)
            modal = representations.modal_residual_embeddings
            corrections = self.joint(deltas) if self.joint_enabled else {
                name: torch.zeros_like(modal[name]) for name in EXPERT_ORDER
            }
            bank = normalized_joint_bank(modal, corrections)
            fused = torch.cat((baseline, bank * baseline.detach().norm(dim=1, keepdim=True)), dim=1)
            self.last_joint_stats = {
                "joint_enabled": int(self.joint_enabled), "joint_sequence_tokens": 1152,
                "joint_token_input_requires_grad": int(deltas.requires_grad),
                "joint_correction_abs_mean": float(torch.stack([
                    value.detach().float().abs().mean() for value in corrections.values()
                ]).mean()),
                "joint_fused_change_abs_max": float((fused.detach() - output.fused_embedding.detach()).abs().max()),
            }
            return replace(output, fused_embedding=fused)

        # Only frozen tail blocks are activation-checkpointed in V8. These final
        # role operators execute once, so captures retain the real training graph.
        with self.baseline.register_forward_hook(field_hook), \
             self.encoder.cnn_stages[-1].register_forward_hook(cnn_hook), \
             self.encoder.transformer_stages[-1].register_forward_hook(transformer_hook), \
             self.encoder.mamba_stages[-1].register_forward_hook(mamba_hook), \
             self.fusion.register_forward_hook(fusion_hook):
            return super().forward(batch, targets, return_aux, retrieval_output)
