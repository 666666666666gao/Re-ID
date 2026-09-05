"""Identity-aware cross-modal supervision for the existing three expert spaces."""
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from .state import EXPERT_ORDER


def cross_modal_identity_loss(
    modal_embeddings: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Average all same-identity positives over six directed modality pairs per expert.

    The caller disables autocast for this FP32 cosine/log-softmax computation.
    No cross-expert alignment or extra inference parameters are introduced.
    """
    positive = labels[:, None].eq(labels[None, :]).float()
    target = positive / positive.sum(dim=1, keepdim=True)
    losses = []
    for expert in EXPERT_ORDER:
        value = F.normalize(modal_embeddings[expert].float(), dim=-1)
        assert value.shape[:2] == (len(labels), 3)
        for query_modality in range(3):
            for gallery_modality in range(3):
                if query_modality == gallery_modality:
                    continue
                logits = value[:, query_modality] @ value[:, gallery_modality].T / temperature
                losses.append(-(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean())
    return torch.stack(losses).mean()
