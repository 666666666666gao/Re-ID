"""Protocol-defined Smooth-AP for the registered MSVR310 source comparison.

Same-ID/same-scene candidates are ignored, never relabelled as negatives.
The original all-identity objective remains in tools.msvr_smooth_ap.
"""
import torch
import torch.nn.functional as F

from tools.msvr_smooth_ap import paired_objectives, TAU


def cross_scene_ap_from_distances(current, historical, identities,
                                  history_identities, scenes, history_scenes):
    """Return eligible-anchor mean loss, per-anchor AP, and positive counts.

AP=0 is a storage placeholder for ineligible anchors; it is excluded from
the loss mean. Whole batches without eligible anchors actually occur in the
registered source sequence and return a graph-connected exact zero.
"""
    batch = current.shape[0]
    assert current.shape == (batch, batch) and historical.shape[0] == batch
    assert current.dtype in (torch.float32, torch.float64)
    assert historical.dtype == current.dtype and historical.device == current.device
    ids = torch.as_tensor(identities, device=current.device)
    mids = torch.as_tensor(history_identities, device=current.device, dtype=ids.dtype)
    env = torch.as_tensor(scenes, device=current.device)
    menv = torch.as_tensor(history_scenes, device=current.device, dtype=env.dtype)
    assert len(ids) == len(env) == batch
    assert len(mids) == len(menv) == historical.shape[1]
    with torch.autocast(current.device.type, enabled=False):
        distance = torch.cat((current, historical), dim=1)
        score = 1 - distance.square() / 2
        same_id = ids[:, None] == torch.cat((ids, mids))[None, :]
        same_scene = env[:, None] == torch.cat((env, menv))[None, :]
        positive = same_id & ~same_scene
        valid = ~(same_id & same_scene)
        counts = positive.sum(1)
        eligible = counts > 0
        assert (~same_id).any(1).all()
        positions = torch.arange(distance.shape[1], device=current.device)
        aps = []
        for i in range(batch):
            pos = positions[positive[i]]
            if pos.numel() == 0:
                aps.append(score[i].sum() * 0)
                continue
            comparisons = torch.sigmoid((score[i][None, :] - score[i, pos][:, None]) / TAU)
            comparison_valid = valid[i][None, :] & (positions[None, :] != pos[:, None])
            comparisons = comparisons.masked_fill(~comparison_valid, 0)
            positive_rank = 1 + (comparisons * positive[i][None, :]).sum(1)
            all_rank = 1 + comparisons.sum(1)
            aps.append((positive_rank / all_rank).mean())
        per_anchor = torch.stack(aps)
        loss = 1 - per_anchor[eligible].mean() if bool(eligible.any()) else distance.sum() * 0
        return loss, per_anchor, counts


def objectives(current, historical, identities, history_identities, scenes, history_scenes):
    """Same three scalar definitions for current, historical leaf, and full graph."""
    hard, standard, standard_ap = paired_objectives(current, historical, identities, history_identities)
    cross, cross_ap, counts = cross_scene_ap_from_distances(
        current, historical, identities, history_identities, scenes, history_scenes)
    return hard, standard, cross, standard_ap, cross_ap, counts


def cross_environment_top1_from_distances(current, historical, identities,
                                          history_identities, environments, history_environments,
                                          tau=0.01):
    """Softplus on best legal positive versus strongest negative, for current anchors."""
    batch = current.shape[0]
    assert current.shape == (batch, batch) and historical.shape[0] == batch
    ids = torch.as_tensor(identities, device=current.device)
    mids = torch.as_tensor(history_identities, device=current.device, dtype=ids.dtype)
    env = torch.as_tensor(environments, device=current.device)
    menv = torch.as_tensor(history_environments, device=current.device, dtype=env.dtype)
    with torch.autocast(current.device.type, enabled=False):
        score = 1 - torch.cat((current, historical), dim=1).float().square() / 2
        same_id = ids[:, None] == torch.cat((ids, mids))[None, :]
        same_env = env[:, None] == torch.cat((env, menv))[None, :]
        positive = same_id & ~same_env
        negative = ~same_id
        eligible = positive.any(dim=1)
        assert negative.any(dim=1).all()
        if not bool(eligible.any()):
            return score.sum() * 0, eligible.sum()
        best_positive = score.masked_fill(~positive, -torch.inf).max(dim=1).values[eligible]
        best_negative = score.masked_fill(~negative, -torch.inf).max(dim=1).values[eligible]
        loss = (tau * F.softplus((best_negative - best_positive) / tau)).mean()
        return loss, eligible.sum()
