"""Read-only global relationship gradients at actual CLIP block inputs.

These are sensitivity proxies, not evidence masks or an identity-learning loss.
"""
import torch
import torch.nn.functional as F


class GlobalInputTap:
    def __init__(self, backbone):
        assert backbone.layers == (3, 7, 11)
        self.blocks = backbone.signal.clip_vision_encoder.base.transformer.resblocks
        self.states = {index: [] for index in backbone.layers}
        self.handles = []

    def __enter__(self):
        for index in self.states:
            def capture(_module, inputs, *, layer=index):
                state = inputs[0]
                assert state.ndim == 3 and state.shape[0] == 129
                assert state.requires_grad and len(self.states[layer]) < 3
                self.states[layer].append(state)
            self.handles.append(self.blocks[index].register_forward_pre_hook(capture))
        return self

    def __exit__(self, _kind, _error, _trace):
        for handle in self.handles:
            handle.remove()

    def ordered_states(self):
        assert all(len(states) == 3 for states in self.states.values())
        return [state for states in self.states.values() for state in states]


def supported_global_target(global_feature, labels, environments):
    """Mean legal positive margins, with the support decision stopped.

One batch target includes query, positive and negative contributions. It is not
query-only attribution. No legal support remains a connected zero; the caller
must report it, never rank tied zero attribution as an evidence mask.
"""
    with torch.autocast(global_feature.device.type, enabled=False):
        unit = F.normalize(global_feature.float(), dim=1)
        score = unit @ unit.T
        same = labels[:, None] == labels[None, :]
        positives = same & (environments[:, None] != environments[None, :])
        legal = positives[:, :, None] & (~same)[:, None, :]
        margin = score[:, :, None] - score[:, None, :]
        supported = legal & (margin.detach() > 0)
        count = supported.sum()
        target = (margin * supported).sum() / count.clamp_min(1)
        involved = (supported.any((1, 2)) | supported.any((0, 2))
                    | supported.any((0, 1)))
    return target, {'legal_positive_pairs': int(positives.sum()),
                    'legal_relations': int(legal.sum()),
                    'supported_relations': int(count),
                    'involved_samples': involved.detach().cpu().tolist()}


def patch_sensitivity(states, gradients):
    """Signed first-order decrease under a small scaling-down of each patch.

Full zeroing is a finite intervention and is not certified by this derivative.
Shape: depth, modality, batch, patch. No min-max confidence or thresholding.
"""
    assert len(states) == len(gradients) == 9
    dot, norms = [], []
    for state, gradient in zip(states, gradients):
        assert gradient is not None and bool(torch.isfinite(gradient).all())
        value = state[1:].detach().float()
        grad = gradient[1:].detach().float()
        dot.append((value * grad).sum(-1).T.cpu())
        norms.append(grad.norm(dim=-1).T.cpu())
    return (torch.stack(dot).reshape(3, 3, -1, 128),
            torch.stack(norms).reshape(3, 3, -1, 128))
