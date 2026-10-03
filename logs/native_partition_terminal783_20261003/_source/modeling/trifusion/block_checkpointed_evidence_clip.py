"""Recompute only original visual block internals; keep outer adapter hooks."""
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint


def checkpointed_block_forward(block, *args, **kwargs):
    if block.training and torch.is_grad_enabled():
        return checkpoint(block._evidence_original_forward, *args,
                          use_reentrant=False, preserve_rng_state=True, **kwargs)
    return block._evidence_original_forward(*args, **kwargs)


def checkpoint_visual_blocks(backbone):
    for block in backbone.signal.clip_vision_encoder.base.transformer.resblocks:
        block._evidence_original_forward = block.forward
        block.forward = MethodType(checkpointed_block_forward, block)
