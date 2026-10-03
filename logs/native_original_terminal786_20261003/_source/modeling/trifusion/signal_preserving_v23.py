"""Modality-specific residual adaptation of the frozen V8 semantic tail."""

from torch import nn
import torch
import torch.nn.functional as F

from .signal_preserving_v8 import ExpertFormationRepresentations
from .state import EXPERT_ORDER, MODALITY_ORDER


class SpectralStageAdapterV23(nn.Module):
    """Apply one zero-output residual MLP to each known spectral modality."""

    def __init__(self, width: int, adapter_width: int) -> None:
        super().__init__()
        self.modal_adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(width, adapter_width), nn.ReLU(),
                          nn.Linear(adapter_width, width))
            for _ in MODALITY_ORDER
        ])
        for adapter in self.modal_adapters:
            nn.init.kaiming_uniform_(adapter[0].weight)
            nn.init.zeros_(adapter[0].bias)
            nn.init.zeros_(adapter[2].weight)
            nn.init.zeros_(adapter[2].bias)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            sequence[:, modality] + adapter(sequence[:, modality])
            for modality, adapter in enumerate(self.modal_adapters)
        ], dim=1)


class SpectralSemanticTailEncoderV23(nn.Module):
    """Share each spectral adapter across roles while preserving their operators."""

    def __init__(self, roles: nn.Module, *, adapter_width: int, train_adapters: bool) -> None:
        super().__init__()
        self.roles = roles
        self.spectral_stages = nn.ModuleList([
            SpectralStageAdapterV23(roles.semantic_width, adapter_width)
            for _ in roles.tail_layer_indices
        ])
        for parameter in self.spectral_stages.parameters():
            parameter.requires_grad_(train_adapters)

    def forward(self, anchor_sequence, reference_sequence):
        roles = self.roles
        batch_size, modality_count = anchor_sequence.shape[:2]
        runtimes = {expert: anchor_sequence for expert in EXPERT_ORDER}
        for stage_index, (block, layer_index) in enumerate(
            zip(roles._tail_blocks, roles.tail_layer_indices, strict=True)
        ):
            for expert in EXPERT_ORDER:
                sequence = roles._run_tail_block(
                    block, roles._to_lbd(runtimes[expert]), layer_index,
                )
                sequence = roles._from_lbd(
                    sequence, batch_size=batch_size, modality_count=modality_count,
                )
                runtimes[expert] = self.spectral_stages[stage_index](sequence)

            cnn = runtimes["cnn"]
            patches = roles.cnn_stages[stage_index](cnn[:, :, 1:].flatten(0, 1))
            runtimes["cnn"] = torch.cat((
                cnn[:, :, :1],
                patches.reshape(batch_size, modality_count, -1, roles.semantic_width),
            ), dim=2)
            transformer = roles.transformer_stages[stage_index](
                runtimes["transformer"].flatten(0, 1),
            )
            runtimes["transformer"] = transformer.reshape_as(runtimes["transformer"])
            mamba = runtimes["mamba"]
            patches = roles.mamba_stages[stage_index](mamba[:, :, 1:])
            runtimes["mamba"] = torch.cat((mamba[:, :, :1], patches), dim=2)

        deltas = {expert: runtimes[expert] - reference_sequence for expert in EXPERT_ORDER}
        modal = {
            "cnn": roles.cnn_head(deltas["cnn"][:, :, 1:]),
            "transformer": roles.transformer_head(deltas["transformer"][:, :, 0]),
            "mamba": roles.mamba_head(deltas["mamba"][:, :, 1:].mean(dim=2)),
        }
        modal = {expert: F.normalize(value, dim=-1) for expert, value in modal.items()}
        return ExpertFormationRepresentations(
            residual_embeddings={expert: F.normalize(value.flatten(1), dim=1)
                                 for expert, value in modal.items()},
            modal_residual_embeddings=modal,
        )
