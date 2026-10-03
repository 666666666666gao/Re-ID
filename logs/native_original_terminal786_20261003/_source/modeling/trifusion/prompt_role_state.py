"""Role prompts carried through, or reset at, frozen CLIP visual blocks."""

import torch
from torch import nn

from .correspondence_context_identity import ContextIdentityTriFusion
from .correspondence_roles import CrossLayerAdaptedCLIP
from .state import MODALITY_ORDER


class PromptRoleBackbone(CrossLayerAdaptedCLIP):
    def __init__(self, signal: nn.Module, adapters: nn.ModuleList, *, prompt_mode: str):
        assert prompt_mode in ("reset", "carry")
        super().__init__(signal, enabled=True)
        self.adapters = adapters
        self.prompt_mode = prompt_mode
        self.prompt_bank = nn.Parameter(torch.empty(3, 4, 768))
        nn.init.normal_(self.prompt_bank, std=768 ** -0.5)
        self.prompt_norm = nn.LayerNorm(768)
        self.prompt_gain = nn.Parameter(torch.full((3,), 0.1))

    def forward(self, images: dict[str, torch.Tensor], camera_ids: torch.Tensor):
        blocks = self.signal.clip_vision_encoder.base.transformer.resblocks
        captured: dict[int, list[torch.Tensor]] = {index: [] for index in self.layers}
        prompt_state = None
        handles = []
        for index, block in enumerate(blocks):
            def add_prompts(_module, inputs, *, layer=index):
                nonlocal prompt_state
                tokens = inputs[0]
                assert tokens.shape[0] == 129
                if layer == 0 or self.prompt_mode == "reset":
                    prompts = self.prompt_bank.reshape(12, 1, 768).expand(-1, tokens.shape[1], -1)
                    prompts = prompts.to(dtype=tokens.dtype)
                else:
                    assert prompt_state is not None
                    prompts = prompt_state
                return (torch.cat((tokens, prompts), dim=0), *inputs[1:])

            def take_prompts(_module, _inputs, output):
                nonlocal prompt_state
                assert output.shape[0] == 141
                prompt_state = output[129:]
                return output[:129]

            handles.append(block.register_forward_pre_hook(add_prompts))
            handles.append(block.register_forward_hook(take_prompts))

        for stage_index, index in enumerate(self.layers):
            def capture(_module, _inputs, output, *, layer=index, stage=stage_index):
                assert prompt_state is not None
                memory = prompt_state.reshape(3, 4, output.shape[1], 768).mean(dim=1)
                memory = self.prompt_norm(memory)
                deltas = [adapter(output) for adapter in self.adapters[stage]]
                captured[layer].append(torch.stack([
                    (output + delta + self.prompt_gain[role] * memory[role][None]).permute(1, 0, 2)
                    for role, delta in enumerate(deltas)
                ], dim=0))
                return output + sum(deltas) / 3
            handles.append(blocks[index].register_forward_hook(capture))

        globals_by_modal = []
        try:
            for name in MODALITY_ORDER:
                _, global_feature = self.signal.clip_vision_encoder(
                    images[name], cam_label=camera_ids, view_label=None,
                )
                globals_by_modal.append(global_feature)
        finally:
            for handle in handles:
                handle.remove()
        assert all(len(captured[index]) == 3 for index in self.layers)
        stages = torch.stack(
            [torch.stack(captured[index], dim=2) for index in self.layers], dim=0,
        )
        return stages, torch.cat(globals_by_modal, dim=1)


class PromptRoleTriFusion(ContextIdentityTriFusion):
    def __init__(self, *args, prompt_mode: str, **kwargs):
        assert kwargs["query_mode"] == "context" and kwargs["auxiliary_target"] == "none"
        super().__init__(*args, **kwargs)
        previous = self.backbone
        self.backbone = PromptRoleBackbone(
            previous.signal, previous.adapters, prompt_mode=prompt_mode,
        )
        self.prompt_mode = prompt_mode
