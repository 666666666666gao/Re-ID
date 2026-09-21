"""Independent Smooth-AP formula adapter, not a novel loss or a trained model.

Brown et al., ECCV 2020, arXiv:2007.12163v2, equations 2/3/6 and section 5.3.
Only current rows are anchors. Historical columns retain caller-defined grads.
"""
import torch

TAU = 0.01


def paired_objectives(current, historical, identities, history_identities):
    """Original split-max/min hard objective and the fixed Smooth-AP objective."""
    batch = current.shape[0]
    ids = torch.as_tensor(identities, device=current.device)
    mids = torch.as_tensor(history_identities, device=current.device, dtype=ids.dtype)
    positive = (ids[:, None] == ids[None, :]) & ~torch.eye(batch, device=current.device, dtype=torch.bool)
    negative = ids[:, None] != ids[None, :]
    assert positive.any(1).all() and negative.any(1).all()
    hp = current.masked_fill(~positive, -torch.inf).max(1).values
    hn = current.masked_fill(~negative, torch.inf).min(1).values
    if historical.shape[1]:
        mp = ids[:, None] == mids[None, :]
        hp = torch.maximum(hp, historical.masked_fill(~mp, -torch.inf).max(1).values)
        hn = torch.minimum(hn, historical.masked_fill(mp, torch.inf).min(1).values)
    hard = torch.nn.functional.relu(hp - hn + .3).mean()
    smooth, per_anchor = smooth_ap_from_distances(current, historical, identities, history_identities)
    return hard, smooth, per_anchor


def smooth_ap_from_distances(current, historical, identities, history_identities):
    """Return mean 1-AP and per-anchor smoothed AP from unit Euclidean distances.

All true positive candidate positions except the current self are included.
Other repeated views remain distinct, matching the existing training contract.
This is training-set approximate AP, not an official evaluation metric.
"""
    batch = current.shape[0]
    assert current.shape == (batch, batch) and historical.shape[0] == batch
    assert current.dtype in (torch.float32, torch.float64)
    assert historical.dtype == current.dtype and historical.device == current.device
    ids = torch.as_tensor(identities, device=current.device)
    mids = torch.as_tensor(history_identities, device=current.device, dtype=ids.dtype)
    assert len(ids) == batch and len(mids) == historical.shape[1]
    with torch.autocast(current.device.type, enabled=False):
        distance = torch.cat((current, historical), dim=1)
        score = 1 - distance.square() / 2
        candidate_ids = torch.cat((ids, mids))
        positions = torch.arange(len(candidate_ids), device=current.device)
        positive = ids[:, None] == candidate_ids[None, :]
        positive[torch.arange(batch, device=current.device), torch.arange(batch, device=current.device)] = False
        assert positive.any(1).all() and (ids[:, None] != candidate_ids[None, :]).any(1).all()
        aps = []
        for i in range(batch):
            pos = positions[positive[i]]
            comparisons = torch.sigmoid((score[i][None, :] - score[i, pos][:, None]) / TAU)
            valid = (positions[None, :] != i) & (positions[None, :] != pos[:, None])
            comparisons = comparisons.masked_fill(~valid, 0)
            positive_rank = 1 + (comparisons * positive[i][None, :]).sum(1)
            all_rank = 1 + comparisons.sum(1)
            aps.append((positive_rank / all_rank).mean())
        per_anchor = torch.stack(aps)
        return 1 - per_anchor.mean(), per_anchor
