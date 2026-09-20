"""Synthetic formula checks only; no data/model or optimizer execution."""
import json
import math
import time

import numpy as np
import torch

from tools.msvr_smooth_ap import smooth_ap_from_distances, paired_objectives


def reference(values, identities, candidates):
    result = []
    for i, row in enumerate(values):
        pos = [j for j, y in enumerate(candidates) if y == identities[i] and j != i]
        allpos = [j for j in range(len(candidates)) if j != i]
        scores = 1 - row*row/2
        terms = []
        for p in pos:
            rp = 1 + sum(1/(1+math.exp(-float(scores[j]-scores[p])/.01)) for j in pos if j != p)
            ra = 1 + sum(1/(1+math.exp(-float(scores[j]-scores[p])/.01)) for j in allpos if j != p)
            terms.append(rp/ra)
        result.append(sum(terms)/len(terms))
    return np.array(result)


def checks():
    started = time.monotonic()
    ids, mids = [0, 0, 1, 1], [0, 1]
    d = torch.tensor([[0,.94,.99,1.02,.98,1.04], [.94,0,1.03,.97,1.01,.99],
                      [.99,1.03,0,.96,1.04,.98], [1.02,.97,.96,0,.99,1.01]], dtype=torch.float64)
    errors = []
    from tools.msvr_role_set_relations import relation_objectives
    for count in (0, 2):
        x = d[:, :4].clone().requires_grad_();y = d[:, 4:4+count].clone().requires_grad_()
        hids = mids[:count]
        hard, loss, ap = paired_objectives(x, y, ids, hids)
        ref = reference(d[:, :4+count].numpy(), ids, ids+hids)
        error = float(np.max(np.abs(ap.detach().numpy()-ref)));errors.append(error)
        assert error < 1e-12
        assert torch.autograd.gradcheck(lambda a,b: smooth_ap_from_distances(a,b,ids,hids)[0],
                                        (x,y), eps=1e-6, atol=1e-6, rtol=1e-4)
        gx, gy = torch.autograd.grad(loss, (x,y), retain_graph=True)
        assert torch.isfinite(gx).all() and torch.isfinite(gy).all() and torch.count_nonzero(gx.diag()) == 0
        original, _, _ = relation_objectives(x, y, ids, hids, [torch.cat((x,y),1)]*3)
        assert torch.equal(hard, original)
        first = torch.autograd.grad(hard, (x,y), retain_graph=True, allow_unused=True)
        second = torch.autograd.grad(original, (x,y), allow_unused=True)
        assert all(a is None and b is None or a is not None and b is not None and torch.equal(a,b)
                   for a,b in zip(first,second,strict=True))
        perm = torch.tensor([2,3,0,1]);hp = torch.tensor(list(reversed(range(count))), dtype=torch.long)
        ll, _ = smooth_ap_from_distances(x.detach()[perm][:,perm], y.detach()[perm][:,hp],
                                         [ids[i] for i in perm], [hids[i] for i in hp])
        assert torch.allclose(loss, ll, atol=1e-12, rtol=0)
    tie = torch.ones(4,4,dtype=torch.float64);tie.fill_diagonal_(0)
    _, ap = smooth_ap_from_distances(tie,torch.empty(4,0,dtype=torch.float64),ids,[])
    assert torch.allclose(ap,torch.full((4,),.5,dtype=torch.float64),atol=1e-12,rtol=0)
    gen = torch.Generator().manual_seed(42)
    z = torch.nn.functional.normalize(torch.randn(128,32,generator=gen),dim=1)
    loss, ap = smooth_ap_from_distances(torch.cdist(z[:64],z[:64]),torch.cdist(z[:64],z[64:]),
                                       [i//8 for i in range(64)],[i//8 for i in range(64)])
    assert torch.isfinite(loss) and ((ap>0)&(ap<=1)).all()
    assert not torch.cuda.is_initialized()
    return dict(status='PASS_SYNTHETIC_SMOOTH_AP_MATH',maximum_scalar_reference_error=max(errors),
                finite_difference_checks=2,original_hard_value_gradient_checks=2,permutation_checks=2,
                self_distance_gradient_zero=True,tie_ap=.5,synthetic_large_shape=[64,128],
                elapsed_seconds=time.monotonic()-started,model_forwards=0,optimizer_updates=0,dataset_reads=0)


if __name__ == '__main__':
    torch.set_num_threads(4)
    print(json.dumps(checks()))
