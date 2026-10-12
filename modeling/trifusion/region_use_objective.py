"""Training-only supervision of actual region comparisons; no novelty claim."""
import torch
import torch.nn.functional as F


def region_similarities(regions):
    assert regions.shape[1:] == (3, 3, 16, 128)
    units = F.normalize(regions.float().reshape(len(regions), 9, 16, 128), dim=-1)
    groups = []
    for group in range(9):
        local = torch.einsum('ipc,jqc->ijpq', units[:, group], units[:, group])
        groups.append(0.5*(local.max(-1).values.mean(-1)+local.max(-2).values.mean(-1)))
    return torch.stack(groups).mean(0)


def region_use_loss(regions, global_feature, labels, environments, *, ambiguity_weighted):
    with torch.autocast(regions.device.type, enabled=False):
        global_unit = F.normalize(global_feature.detach().float(), dim=1)
        global_score = global_unit @ global_unit.T
        region_score = region_similarities(regions)
        same = labels[:, None] == labels[None, :]
        positives = same & (environments[:, None] != environments[None, :])
        legal = positives[:, :, None] & (~same)[:, None, :]
        global_margin = global_score[:, :, None]-global_score[:, None, :]
        region_margin = region_score[:, :, None]-region_score[:, None, :]
        scale = regions.shape[-1]**0.5
        weights = torch.sigmoid(-scale*global_margin) if ambiguity_weighted else torch.ones_like(global_margin)
        weights = weights*legal
        mass = weights.sum((1, 2))
        losses = F.softplus(-scale*0.5*(global_margin+region_margin))
        # The sealed prior source probe measured empty legal-positive batches.
        # Such queries contribute zero; ordinary author tasks remain active.
        per_query = (weights*losses).sum((1, 2))/mass.clamp_min(1e-12)
        eligible = positives.any(1)
        loss = per_query.sum()/eligible.sum().clamp_min(1)
    return loss, {'eligible_queries': int(eligible.sum()),
                  'legal_positive_pairs': int(positives.sum()),
                  'legal_triplets': int(legal.sum()),
                  'global_wrong_triplets': int((legal & (global_margin <= 0)).sum()),
                  'ambiguity_weighted': ambiguity_weighted}
