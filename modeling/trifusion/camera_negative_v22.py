"""Source-camera negative ordering from the MCNL loss (AAAI 2020)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def camera_pair_support(labels: torch.Tensor, cameras: torch.Tensor):
    same_identity = labels[:, None].eq(labels[None, :])
    same_camera = cameras[:, None].eq(cameras[None, :])
    diagonal = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    positives = same_identity & ~diagonal
    same_negatives = ~same_identity & same_camera
    other_negatives = ~same_identity & ~same_camera
    valid = positives.any(dim=1) & same_negatives.any(dim=1) & other_negatives.any(dim=1)
    assert bool(valid.any()), "MCNL requires observed positive and both negative-camera groups"
    stats = {
        "valid_rows": int(valid.sum()),
        "same_negative_missing_rows": int((~same_negatives.any(dim=1)).sum()),
        "other_negative_missing_rows": int((~other_negatives.any(dim=1)).sum()),
        "cross_camera_positive_rows": int((positives & ~same_camera).any(dim=1).sum()),
    }
    return positives, same_negatives, other_negatives, valid, stats


def multi_camera_negative_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    cameras: torch.Tensor,
    *,
    positive_margin: float = 0.1,
    camera_margin: float = 0.1,
):
    """Mean over rows with both true-negative groups; all true ID positives are kept."""
    positives, same_negatives, other_negatives, valid, stats = camera_pair_support(labels, cameras)
    unit = F.normalize(embeddings.float(), dim=1)
    distances = torch.cdist(unit, unit, p=2)
    positive = distances.masked_fill(~positives, -torch.inf).max(dim=1).values[valid]
    same_negative = distances.masked_fill(~same_negatives, torch.inf).min(dim=1).values[valid]
    other_negative = distances.masked_fill(~other_negatives, torch.inf).min(dim=1).values[valid]
    positive_term = F.relu(positive_margin + positive - other_negative)
    camera_term = F.relu(camera_margin + other_negative - same_negative)
    loss = (positive_term + camera_term).mean()
    return loss, {
        **stats,
        "positive_term": float(positive_term.detach().mean()),
        "camera_term": float(camera_term.detach().mean()),
        "positive_active_rows": int((positive_term > 0).sum()),
        "camera_active_rows": int((camera_term > 0).sum()),
    }
