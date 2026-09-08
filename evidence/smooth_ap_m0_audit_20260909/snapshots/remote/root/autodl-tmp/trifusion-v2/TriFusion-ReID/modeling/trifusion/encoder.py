"""Public three-expert encoder seam."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .interventions import FullNetworkIntervention
from .state import (
    EXPERT_ORDER,
    MODALITY_ORDER,
    ExpertState,
    ExpertStateMap,
    PackedExpertOutput,
)


class TriBranchEncoder(nn.Module):
    """Pack valid modalities and run every configured heterogeneous expert."""

    def __init__(
        self,
        experts: Mapping[str, nn.Module],
        *,
        tokenizer: nn.Module | None = None,
        reliability_gate: nn.Module | None = None,
        collaborator: nn.Module | None = None,
        refresh_final_reliability: bool = False,
    ) -> None:
        super().__init__()
        if set(experts) != set(EXPERT_ORDER) or len(experts) != len(EXPERT_ORDER):
            raise ValueError(f"experts must contain exactly {EXPERT_ORDER}")
        self.experts = nn.ModuleDict(
            {expert: experts[expert] for expert in EXPERT_ORDER}
        )
        self.tokenizer = tokenizer
        if (reliability_gate is None) != (collaborator is None):
            raise ValueError("reliability_gate and collaborator must be configured together")
        if collaborator is not None and tokenizer is None:
            raise ValueError("deep collaboration requires the shared tokenizer")
        self.reliability_gate = reliability_gate
        self.collaborator = collaborator
        self.refresh_final_reliability = bool(refresh_final_reliability)

    def forward(
        self,
        images: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> ExpertStateMap:
        return self._forward(images, modality_mask, intervention=None)

    def _forward_intervened(
        self,
        images: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
        intervention: FullNetworkIntervention,
    ) -> ExpertStateMap:
        return self._forward(images, modality_mask, intervention=intervention)

    def _forward(
        self,
        images: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
        intervention: FullNetworkIntervention | None,
    ) -> ExpertStateMap:
        self._validate_inputs(images, modality_mask)
        all_missing = (~modality_mask).all(dim=1).nonzero(as_tuple=False).flatten()
        if all_missing.numel():
            rows = all_missing.detach().cpu().tolist()
            raise ValueError(f"all-missing modality rows: {rows}")

        stacked_images = torch.stack(
            [images[modality] for modality in MODALITY_ORDER], dim=1
        )
        packed_images = stacked_images[modality_mask]
        if self.tokenizer is None:
            expert_inputs = {expert: packed_images for expert in EXPERT_ORDER}
        else:
            modality_indices = torch.arange(
                len(MODALITY_ORDER), device=modality_mask.device
            ).view(1, -1).expand(modality_mask.shape[0], -1)
            expert_inputs = self.tokenizer(
                packed_images, modality_indices[modality_mask]
            )
        if self.collaborator is not None:
            return self._forward_collaborative(
                expert_inputs,
                modality_mask,
                intervention=intervention,
            )
        if intervention is not None:
            raise RuntimeError("full-network intervention requires deep collaboration")
        states = {
            expert: self._scatter_expert_output(
                self.experts[expert](expert_inputs[expert]),
                modality_mask,
                expert,
            )
            for expert in EXPERT_ORDER
        }
        return ExpertStateMap(states, modality_mask=modality_mask)

    def _forward_collaborative(
        self,
        expert_inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
        intervention: FullNetworkIntervention | None,
    ) -> ExpertStateMap:
        runtimes = {
            expert: self.experts[expert].initialize(expert_inputs[expert])
            for expert in EXPERT_ORDER
        }
        reliability = None
        relay_results = []
        final_states = None
        for stage in (1, 2, 3):
            packed_outputs = {}
            for expert in EXPERT_ORDER:
                runtimes[expert] = self.experts[expert].run_stage(
                    runtimes[expert], stage
                )
                packed_outputs[expert] = self.experts[expert].summarize(
                    runtimes[expert], stage
                )
            stage_states = ExpertStateMap(
                {
                    expert: self._scatter_expert_output(
                        packed_outputs[expert], modality_mask, expert
                    )
                    for expert in EXPERT_ORDER
                },
                modality_mask=modality_mask,
            )
            if stage == 1:
                reliability = self.reliability_gate(stage_states, modality_mask)
            if stage < 3:
                if intervention is None:
                    relay_result = self.collaborator(
                        stage_states, reliability, stage
                    )
                else:
                    relay_result = self.collaborator._forward_intervened(
                        stage_states,
                        reliability,
                        stage,
                        intervention,
                    )
                relay_results.append(relay_result)
                for expert in EXPERT_ORDER:
                    relayed_packed = relay_result.states[expert].tokens[
                        modality_mask
                    ]
                    runtimes[expert] = self.experts[expert].inject(
                        runtimes[expert], relayed_packed
                    )
            else:
                final_states = stage_states
                if self.refresh_final_reliability:
                    reliability = self.reliability_gate(
                        final_states,
                        modality_mask,
                    )

        if reliability is None or final_states is None:
            raise RuntimeError("collaborative encoder did not complete its schedule")
        return ExpertStateMap(
            {expert: final_states[expert] for expert in EXPERT_ORDER},
            modality_mask=modality_mask,
            reliability=reliability,
            relay_results=tuple(relay_results),
        )

    @staticmethod
    def _scatter_expert_output(
        packed: PackedExpertOutput,
        modality_mask: torch.Tensor,
        expert: str,
    ) -> ExpertState:
        batch_size, modality_count = modality_mask.shape
        valid_slot_count = int(modality_mask.sum().item())

        def scatter(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim < 1 or tensor.shape[0] != valid_slot_count:
                raise ValueError("expert output must lead with packed valid slots")
            output = tensor.new_zeros(
                (batch_size, modality_count, *tensor.shape[1:])
            )
            output[modality_mask] = tensor
            return output

        return ExpertState(
            tokens=scatter(packed.tokens),
            global_embedding=scatter(packed.global_embedding),
            private_embedding=scatter(packed.private_embedding),
            role_payload={
                name: scatter(value) for name, value in packed.role_payload.items()
            },
            modality_mask=modality_mask,
            stage=packed.stage,
            expert=expert,
        )

    @staticmethod
    def _validate_inputs(
        images: Mapping[str, torch.Tensor], modality_mask: torch.Tensor
    ) -> None:
        if tuple(images.keys()) != MODALITY_ORDER:
            raise ValueError(f"images must follow modality order {MODALITY_ORDER}")
        if modality_mask.dtype != torch.bool or modality_mask.ndim != 2:
            raise ValueError("modality_mask must be a rank-2 bool tensor")
        if modality_mask.shape[1] != len(MODALITY_ORDER):
            raise ValueError("modality_mask columns must be RGB, NI, TI")
        batch_size = modality_mask.shape[0]
        reference_shape = None
        for modality in MODALITY_ORDER:
            image = images[modality]
            if image.ndim != 4 or image.shape[0] != batch_size:
                raise ValueError("each image tensor must have shape B,C,H,W")
            if image.device != modality_mask.device:
                raise ValueError("images and modality_mask must share one device")
            if reference_shape is None:
                reference_shape = image.shape
            elif image.shape != reference_shape:
                raise ValueError("all modality image tensors must share one shape")


__all__ = ["TriBranchEncoder"]
