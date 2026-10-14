"""A fixed inference parameter for the existing equal-energy residual bank."""
import torch
import torch.nn.functional as F


def apply_residual_budget(embedding, baseline_width, ratio):
    assert embedding.ndim==2 and 0<baseline_width<embedding.shape[1]
    assert 0<ratio<=1
    return F.normalize(torch.cat((embedding[:,:baseline_width],
                                  ratio*embedding[:,baseline_width:]),dim=1),dim=1)


def cached_budget_distance(baseline_distance, equal_energy_fused_distance, ratio):
    """Valid when the original baseline and bank have equal, nonzero norms."""
    assert baseline_distance.shape==equal_energy_fused_distance.shape and 0<ratio<=1
    weight=2*ratio**2/(1+ratio**2)
    return (1-weight)*baseline_distance+weight*equal_energy_fused_distance
