"""Source-labelled ranking responsibility for the existing nine V8 slots.

BIER-inspired research hypothesis; independently implemented, not a BIER port.
The inference model and every original loss remain unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")


def responsibility_loss(output, labels, *, temperature):
    """Average over all ordered, non-self-row true identity triplets and slots.

    Repeated source records remain augmented training exposures, not independent
    identities or observations. Class zero is a valid identity. Camera labels do
    not redefine identity. Weights are constants for this backward pass.
    """
    slots = torch.cat([output.modal_residual_embeddings[e].float() for e in EXPERTS], dim=1)
    norms = slots.norm(dim=-1)
    assert bool((norms > 0).all())
    slots = F.normalize(slots, dim=-1).transpose(0, 1)
    slot_similarity = slots @ slots.transpose(1, 2)
    fused = F.normalize(output.fused_embedding.detach().float(), dim=1)
    baseline = F.normalize(output.baseline_embedding.detach().float(), dim=1)
    fused_similarity = fused @ fused.T
    baseline_similarity = baseline @ baseline.T
    same = labels[:, None].eq(labels[None, :])
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    q, p, n = torch.where(positive[:, :, None] & ~same[:, None, :])
    assert q.numel() > 0
    margins = slot_similarity[:, q, p] - slot_similarity[:, q, n]
    fused_margin = fused_similarity[q, p] - fused_similarity[q, n]
    weights = (torch.sigmoid(-fused_margin / temperature)[None, :]
               * torch.sigmoid(-margins / temperature)).detach()
    penalties = temperature * F.softplus(-margins / temperature)
    loss = (weights * penalties).mean()
    decomposition = 0.5 * baseline_similarity + slot_similarity.detach().sum(dim=0) / 18
    with torch.no_grad():
        stats = {
            "responsibility_triplets": int(q.numel()),
            "responsibility_slot_triplets": int(margins.numel()),
            "responsibility_fused_nonpositive": int((fused_margin <= 0).sum()),
            "responsibility_raw_slot_norm_max_error": float((norms - 1).abs().max()),
            "responsibility_fused_decomposition_max_error": float((fused_similarity - decomposition).abs().max()),
        }
        for index, (expert, modality) in enumerate((e, m) for e in EXPERTS for m in MODALITIES):
            w = weights[index]
            prefix = f"responsibility_{expert}_{modality}_"
            stats.update({
                prefix + "weight_mean": float(w.mean()),
                prefix + "weight_min": float(w.min()),
                prefix + "weight_max": float(w.max()),
                prefix + "effective_sample_size": float(w.sum().square() / w.square().sum()),
                prefix + "slot_nonpositive": int((margins[index] <= 0).sum()),
                prefix + "joint_nonpositive": int(((fused_margin <= 0) & (margins[index] <= 0)).sum()),
                prefix + "weighted_loss": float((w * penalties[index]).mean()),
            })
    return loss, stats


def same_block_gradient_diagnostics(model, fused_loss, role_loss, auxiliary_loss, scaler):
    """Compare three objectives in each SAME encoder parameter coordinate block."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    partitions = {e: [i for i, (n, _) in enumerate(named) if n.startswith(f"encoder.{e}_")] for e in EXPERTS}
    assert [len(partitions[e]) for e in EXPERTS] == [42, 54, 93]
    assert sum(map(len, partitions.values())) == len(named) == 189
    vectors = {}
    for name, objective in (("fused", fused_loss), ("role", role_loss), ("auxiliary", auxiliary_loss)):
        gradients = torch.autograd.grad(scaler.scale(objective), [p for _, p in named], retain_graph=True)
        assert all(bool(torch.isfinite(g).all()) for g in gradients)
        vectors[name] = {e: torch.cat([gradients[i].float().flatten() for i in indices]) / scaler.get_scale()
                         for e, indices in partitions.items()}
    result = {}
    for expert in EXPERTS:
        fused, role, auxiliary = [vectors[name][expert] for name in ("fused", "role", "auxiliary")]
        base = fused + role
        assert all(bool(vector.norm() > 0) for vector in (fused, role, auxiliary, base))
        result[expert] = {
            "parameter_tensors": len(partitions[expert]), "parameter_elements": fused.numel(),
            "fused_gradient_norm": float(fused.norm()), "role_gradient_norm": float(role.norm()),
            "base_gradient_norm": float(base.norm()), "auxiliary_gradient_norm": float(auxiliary.norm()),
            "fused_role_cosine": float(F.cosine_similarity(fused, role, dim=0)),
            "base_auxiliary_cosine": float(F.cosine_similarity(base, auxiliary, dim=0)),
            "auxiliary_to_base_norm_ratio": float(auxiliary.norm() / base.norm()),
            "new_encoder_gradient_nonzero": bool(auxiliary.norm() > 0),
        }
    return result
