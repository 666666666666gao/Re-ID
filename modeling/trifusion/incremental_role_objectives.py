"""Training-only comparisons for fused evidence; no new model parameters.

The batch-ratio control independently expresses MDReID's published distance
ratio idea, adapted to current global/correction/fused components. It is not
the author's full shared/private model or a copy of its unlicensed code.
The repair/keep comparison uses legal source-environment instance relations
and a detached global reference. Neither objective is claimed to be novel.
"""
import torch
import torch.nn.functional as F


def _distances(feature):
    squared = feature.square().sum(dim=1)
    return (squared[:, None] + squared[None, :] - 2 * feature @ feature.T).clamp_min(1e-12).sqrt()


def batch_ratio_loss(global_feature, correction, raw_fused, labels):
    with torch.autocast(raw_fused.device.type, enabled=False):
        same = labels[:, None].eq(labels[None, :])
        assert bool((~same).any(dim=1).all())
        extremes = []
        for feature in (raw_fused.float(), global_feature.detach().float(), correction.detach().float()):
            distance = _distances(feature)
            positive = distance.masked_fill(~same, -torch.inf).amax()
            negative = distance.masked_fill(same, torch.inf).amin()
            extremes.append((positive, negative))
        positive_ratio = extremes[0][0] / sum(pair[0] for pair in extremes)
        negative_ratio = extremes[0][1] / sum(pair[1] for pair in extremes)
        loss = positive_ratio + 1 - negative_ratio
    return loss, dict(incremental_objective='md_batch_ratio_adaptation',
        positive_batch_ratio=float(positive_ratio.detach()), negative_batch_ratio=float(negative_ratio.detach()))


def _query_mean(values, mask):
    counts = mask.sum(dim=(1, 2))
    means = (values * mask).sum(dim=(1, 2)) / counts.clamp_min(1)
    # Empty query/cell support is a measured property of the source batches.
    # Its loss is defined as zero; ordinary author tasks remain active.
    return means.sum() / (counts > 0).sum().clamp_min(1)


def repair_keep_loss(global_feature, fused, labels, environments):
    with torch.autocast(fused.device.type, enabled=False):
        g = F.normalize(global_feature.detach().float(), dim=1)
        f = F.normalize(fused.float(), dim=1)
        same = labels[:, None].eq(labels[None, :])
        positive = same & environments[:, None].ne(environments[None, :])
        legal = positive[:, :, None] & (~same)[:, None, :]
        g_similarity, f_similarity = g @ g.T, f @ f.T
        g_margin = g_similarity[:, :, None] - g_similarity[:, None, :]
        f_margin = f_similarity[:, :, None] - f_similarity[:, None, :]
        repair = legal & (g_margin <= 0)
        keep = legal & (g_margin > 0)
        # Wrong source relations must reach a positive 0.1 cosine margin;
        # already-correct relations must retain their detached global margin.
        repair_loss = _query_mean(F.relu(0.1 - f_margin), repair)
        keep_loss = _query_mean(F.relu(g_margin - f_margin), keep)
        loss = 0.5 * (repair_loss + keep_loss)
    with torch.no_grad():
        stats = dict(incremental_objective='legal_instance_global_repair_keep',
            legal_positive_pairs=int(positive.sum()), legal_triplets=int(legal.sum()),
            eligible_queries=int(positive.any(dim=1).sum()),
            multiple_positive_queries=int((positive.sum(dim=1) >= 2).sum()),
            repair_triplets=int(repair.sum()), keep_triplets=int(keep.sum()),
            repair_queries=int(repair.any(dim=(1, 2)).sum()), keep_queries=int(keep.any(dim=(1, 2)).sum()),
            repair_loss=float(repair_loss), keep_loss=float(keep_loss),
            repair_active_triplets=int((repair & (f_margin < 0.1)).sum()),
            keep_active_triplets=int((keep & (f_margin < g_margin)).sum()))
    return loss, stats
