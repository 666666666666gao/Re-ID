"""Synthetic protocol and gradient checks, no dataset or model execution."""
import json
import math
import time

import numpy as np
import torch

from tools.msvr_cross_scene_smooth_ap import cross_scene_ap_from_distances


def reference(distance, ids, mids, scenes, mscenes):
    candidate_ids, candidate_scenes = ids + mids, scenes + mscenes
    aps, counts = [], []
    for i, row in enumerate(distance):
        positives = [j for j, (y, s) in enumerate(zip(candidate_ids, candidate_scenes))
                     if y == ids[i] and s != scenes[i]]
        valid = [j for j, (y, s) in enumerate(zip(candidate_ids, candidate_scenes))
                 if not (y == ids[i] and s == scenes[i])]
        score = 1 - row * row / 2
        terms = []
        for positive in positives:
            def rank(indices):
                return 1 + sum(1 / (1 + math.exp(-float(score[j] - score[positive]) / .01))
                               for j in indices if j != positive)
            terms.append(rank(positives) / rank(valid))
        aps.append(sum(terms) / len(terms) if terms else 0.)
        counts.append(len(positives))
    eligible = [ap for ap, count in zip(aps, counts) if count]
    return 1 - sum(eligible) / len(eligible) if eligible else 0., np.array(aps), np.array(counts)


def checks():
    started = time.monotonic()
    from tools.check_msvr_smooth_ap_math import checks as standard_checks
    standard = standard_checks()
    ids, mids = [0, 0, 1, 1], [0, 1]
    scenes = [0, 1, 0, 0]
    d = torch.tensor([[0,.94,.99,1.02,.98,1.04], [.94,0,1.03,.97,1.01,.99],
                      [.99,1.03,0,.96,1.04,.98], [1.02,.97,.96,0,.99,1.01]], dtype=torch.float64)
    scalar_errors = []
    for count, mscenes in ((0, []), (2, [0, 1]), (2, [0, 0])):
        x = d[:, :4].clone().requires_grad_()
        y = d[:, 4:4+count].clone().requires_grad_()
        hids = mids[:count]
        loss, ap, counts = cross_scene_ap_from_distances(x, y, ids, hids, scenes, mscenes)
        expected, expected_ap, expected_counts = reference(d[:, :4+count].numpy(), ids, hids, scenes, mscenes)
        error = max(abs(loss.item()-expected), float(np.max(np.abs(ap.detach().numpy()-expected_ap))))
        scalar_errors.append(error)
        assert error < 1e-12 and np.array_equal(counts.numpy(), expected_counts)
        assert torch.autograd.gradcheck(lambda a,b: cross_scene_ap_from_distances(a,b,ids,hids,scenes,mscenes)[0],
                                        (x,y), eps=1e-6, atol=1e-6, rtol=1e-4)
        gx, gy = torch.autograd.grad(loss, (x, y))
        both = torch.cat((gx, gy), 1)
        for i in range(4):
            for j, (identity, scene) in enumerate(zip(ids+hids, scenes+mscenes)):
                if identity == ids[i] and scene == scenes[i]:
                    assert both[i, j] == 0
            if counts[i] == 0:
                assert torch.count_nonzero(both[i]) == 0
        # ID1 has no legal positive in the no-history fixture, but remains
        # a true negative to eligible ID0, including same-scene negatives.
        if count == 0:
            assert counts.tolist() == [1, 1, 0, 0]
            assert both[0, 2] != 0 and both[1, 3] != 0
        if count and mscenes == [0, 1]:
            assert torch.count_nonzero(gy) > 0
        perm = torch.tensor([2,3,0,1]); hp = torch.arange(count-1, -1, -1)
        perm_loss, _, _ = cross_scene_ap_from_distances(
            x.detach()[perm][:,perm], y.detach()[perm][:,hp],
            [ids[i] for i in perm], [hids[i] for i in hp],
            [scenes[i] for i in perm], [mscenes[i] for i in hp])
        assert torch.allclose(loss, perm_loss, rtol=0, atol=1e-12)
    x = d[:, :4].clone().requires_grad_(); y = d[:, 4:].clone().requires_grad_()
    loss, ap, counts = cross_scene_ap_from_distances(x,y,ids,mids,[0]*4,[0]*2)
    gx, gy = torch.autograd.grad(loss, (x,y))
    assert loss.item() == 0 and counts.sum() == 0 and ap.count_nonzero() == 0
    assert gx.count_nonzero() == gy.count_nonzero() == 0
    # Independent tie arithmetic: one cross-scene positive and two negatives
    # give rank+ = 1 and rank = 2, while ignored same-scene self adds nothing.
    tie = torch.ones(4,4,dtype=torch.float64); tie.fill_diagonal_(0)
    loss, ap, counts = cross_scene_ap_from_distances(tie,torch.empty(4,0,dtype=torch.float64),ids,[],[0,1,0,1],[])
    assert torch.equal(ap,torch.full((4,),.5,dtype=torch.float64)) and loss == .5
    assert not torch.cuda.is_initialized()
    return dict(status='PASS_SYNTHETIC_CROSS_SCENE_SMOOTH_AP_MATH', standard=standard,
                maximum_scalar_reference_error=max(scalar_errors),finite_difference_checks=3,
                permutation_checks=3,ignored_same_identity_scene_zero_derivative=True,
                ineligible_anchor_retained_as_negative=True,whole_ineligible_batch_connected_zero=True,
                historical_candidate_derivative_nonzero=True,tie_ap=.5,
                model_forwards=0,optimizer_updates=0,dataset_reads=0,
                elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    torch.set_num_threads(4)
    print(json.dumps(checks()))
