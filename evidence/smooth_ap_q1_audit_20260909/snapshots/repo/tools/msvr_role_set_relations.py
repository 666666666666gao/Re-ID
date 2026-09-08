"""Role-proposed negative relations, optimized only in the fused metric space.

This is a mean-hinge adaptation, not a new loss family. Mining indices are
fixed for each backward pass. The original hard term retains its existing
max/min tie derivative, including the current/history split.
"""
import torch
import torch.nn.functional as F


def relation_objectives(current, historical, identities, history_identities,
                        role_distances, margin=.3):
    """Return original hard loss, role-union mean loss and detached selection.

current/historical are differentiable fused distance matrices, BxB and BxH.
The three role matrices are Bx(B+H), used only to choose negative indices.
Current sample positions (including repeated sampled views) retain the old
training contract; the caller supplies the already-filtered history queue.
"""
    batch, count = historical.shape
    assert current.shape == (batch, batch) and len(role_distances) == 3
    ids = torch.as_tensor(identities, device=current.device)
    mids = torch.as_tensor(history_identities, device=current.device, dtype=ids.dtype)
    assert len(ids) == batch and len(mids) == count
    positive = (ids[:, None] == ids[None, :]) & ~torch.eye(batch, device=current.device, dtype=torch.bool)
    negative = ids[:, None] != ids[None, :]
    assert positive.any(1).all() and negative.any(1).all()
    hp = current.masked_fill(~positive, -torch.inf).max(1).values
    hn = current.masked_fill(~negative, torch.inf).min(1).values
    mp = ids[:, None] == mids[None, :]
    if count:
        hp = torch.maximum(hp, historical.masked_fill(~mp, -torch.inf).max(1).values)
        hn = torch.minimum(hn, historical.masked_fill(mp, torch.inf).min(1).values)
    distance = torch.cat((current, historical), dim=1)
    negatives = torch.cat((negative, ~mp), dim=1)
    hard = F.relu(hp-hn+margin)
    with torch.no_grad():
        first = distance.detach().masked_fill(~negatives, torch.inf).argmin(1)
        proposals = [first]
        for role in role_distances:
            assert role.shape == distance.shape and torch.isfinite(role).all()
            proposals.append(role.detach().masked_fill(~negatives, torch.inf).argmin(1))
        proposals = torch.stack(proposals, dim=1)
        selected = torch.zeros_like(negatives)
        selected.scatter_(1, proposals, True)
        assert (selected & ~negatives).sum() == 0
        extra = selected.clone()
        extra.scatter_(1, first[:, None], False)
        counts = selected.sum(1)
    hinges = F.relu(hp[:, None]-distance+margin)
    # Keep the old hard term rather than changing its tie-gradient convention.
    combined = (hard+(hinges*extra).sum(1))/counts
    return hard.mean(), combined.mean(), dict(
        proposals=proposals, selected=selected, extra=extra,
        counts=counts, extra_active=(hinges.detach()>0) & extra,
        per_anchor_hard=hard.detach(), per_anchor_candidate=combined.detach(),
    )


def fused_distances(features, history):
    """The same FP32 metric path as the existing history-gradient objective."""
    with torch.autocast(features.device.type, enabled=False):
        unit = F.normalize(features.float(), dim=1)
        return torch.cdist(unit, unit), torch.cdist(unit, history.float())
