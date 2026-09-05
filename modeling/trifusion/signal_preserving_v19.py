"""Give each V8 role its own CLIP semantic tail, preserving the Signal path."""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from .signal_preserving_v8 import ExpertFormationRepresentations
from .state import EXPERT_ORDER


class PrivateSemanticTailEncoderV19(nn.Module):
    """Reuse loaded role modules and replace only their shared tail dispatch."""

    def __init__(self, roles: nn.Module, *, train_private_tail: bool) -> None:
        super().__init__()
        self.roles = roles
        self.private_tails = nn.ModuleDict({
            expert: nn.ModuleList([deepcopy(block) for block in roles._tail_blocks])
            for expert in EXPERT_ORDER
        })
        for parameter in self.private_tails.parameters():
            parameter.requires_grad_(train_private_tail)

    def forward(self, anchor_sequence, reference_sequence):
        roles = self.roles
        batch_size, modality_count = anchor_sequence.shape[:2]
        runtimes = {expert: anchor_sequence for expert in EXPERT_ORDER}
        for stage_index, layer_index in enumerate(roles.tail_layer_indices):
            for expert in EXPERT_ORDER:
                sequence = roles._to_lbd(runtimes[expert])
                sequence = roles._run_tail_block(
                    self.private_tails[expert][stage_index], sequence, layer_index,
                )
                runtimes[expert] = roles._from_lbd(
                    sequence, batch_size=batch_size, modality_count=modality_count,
                )

            cnn = runtimes["cnn"]
            cnn_patches = roles.cnn_stages[stage_index](cnn[:, :, 1:].flatten(0, 1))
            runtimes["cnn"] = torch.cat((
                cnn[:, :, :1],
                cnn_patches.reshape(batch_size, modality_count, -1, roles.semantic_width),
            ), dim=2)

            transformer = roles.transformer_stages[stage_index](
                runtimes["transformer"].flatten(0, 1),
            )
            runtimes["transformer"] = transformer.reshape_as(runtimes["transformer"])

            mamba = runtimes["mamba"]
            mamba_patches = roles.mamba_stages[stage_index](mamba[:, :, 1:])
            runtimes["mamba"] = torch.cat((mamba[:, :, :1], mamba_patches), dim=2)

        deltas = {
            expert: runtimes[expert] - reference_sequence for expert in EXPERT_ORDER
        }
        modal = {
            "cnn": roles.cnn_head(deltas["cnn"][:, :, 1:]),
            "transformer": roles.transformer_head(deltas["transformer"][:, :, 0]),
            "mamba": roles.mamba_head(deltas["mamba"][:, :, 1:].mean(dim=2)),
        }
        modal = {expert: F.normalize(value, dim=-1) for expert, value in modal.items()}
        return ExpertFormationRepresentations(
            residual_embeddings={
                expert: F.normalize(value.flatten(1), dim=1)
                for expert, value in modal.items()
            },
            modal_residual_embeddings=modal,
        )


def private_tail_storage_is_disjoint(encoder: PrivateSemanticTailEncoderV19) -> bool:
    """Check real storage addresses, including all three original Signal blocks."""
    original = {
        p.untyped_storage().data_ptr()
        for block in encoder.roles._tail_blocks for p in block.parameters()
    }
    private = [p.untyped_storage().data_ptr() for p in encoder.private_tails.parameters()]
    return len(private) == len(set(private)) and original.isdisjoint(private)


def optimizer_parameter_groups(model: nn.Module, *, role_lr: float, tail_lr: float):
    tail_ids = {id(p) for p in model.encoder.private_tails.parameters()}
    roles = [p for p in model.parameters() if p.requires_grad and id(p) not in tail_ids]
    tails = [p for p in model.encoder.private_tails.parameters() if p.requires_grad]
    groups = [{"params": roles, "lr": role_lr, "initial_lr": role_lr, "name": "roles_and_heads"}]
    if tails:
        groups.append({"params": tails, "lr": tail_lr, "initial_lr": tail_lr, "name": "private_tail"})
    return groups
