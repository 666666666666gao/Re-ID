"""Independent small CPU graph checks of the imported objective, no real model."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import shutil
import torch
import torch.nn.functional as F

ROOT = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
path = ROOT/'tools/msvr_role_set_relations.py'
module_spec = importlib.util.spec_from_file_location('audited_role_objective', path)
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
torch.set_num_threads(4)
torch.manual_seed(42)
cases = []


def explicit(dc, dh, ids, mids, proposals):
    hard_terms, candidate_terms = [], []
    for i, identity in enumerate(ids):
        positive = torch.stack([dc[i,j] for j,x in enumerate(ids) if x == identity and i != j]).max()
        negative = torch.stack([dc[i,j] for j,x in enumerate(ids) if x != identity]).min()
        hp = [dh[i,j] for j,x in enumerate(mids) if x == identity]
        hn = [dh[i,j] for j,x in enumerate(mids) if x != identity]
        if hp:
            positive = torch.maximum(positive, torch.stack(hp).max())
        if hn:
            negative = torch.minimum(negative, torch.stack(hn).min())
        hard = (positive-negative+.3).clamp_min(0)
        hard_terms.append(hard)
        first = proposals[i][0]
        terms = [hard]
        for j in sorted(set(proposals[i]) - {first}):
            distance = dc[i,j] if j < len(ids) else dh[i,j-len(ids)]
            terms.append((positive-distance+.3).clamp_min(0))
        candidate_terms.append(torch.stack(terms).sum()/len(set(proposals[i])))
    return torch.stack(hard_terms).mean(), torch.stack(candidate_terms).mean()


for label, mids, mode in [('multi_proposal', [0,1,4,5], 'different'),
                         ('single_agree', [0,1,4,5], 'same'),
                         ('no_history', [], 'different'),
                         ('duplicate_extra', [0,1,4,5], 'duplicate'),
                         ('current_history_tie_agree', [0,1,2,3], 'tie'),
                         ('current_history_tie_extra', [0,1,2,3], 'tie_extra')]:
    ids = [0,0,1,1,2,2,3,3]
    dc = (torch.rand(8,8,dtype=torch.float64)*.2+.5).requires_grad_()
    dh = (torch.rand(8,len(mids),dtype=torch.float64)*.2+.5).requires_grad_()
    if mode.startswith('tie'):
        with torch.no_grad():
            dc.fill_(.6); dh.fill_(.6)
    combined = torch.cat((dc,dh),1).detach()
    if mode in ('same','tie'):
        roles = [combined.clone().requires_grad_() for _ in range(3)]
    else:
        roles = [torch.rand_like(combined).requires_grad_() for _ in range(3)]
        if mode == 'duplicate':
            roles = [roles[0]]*3
    old, candidate, selection = module.relation_objectives(dc,dh,ids,mids,roles)
    allids = ids+mids
    proposal = [[min([j for j,x in enumerate(allids) if x != ids[i]], key=lambda j:(float(a[i,j].detach()),j))
                 for a in [combined,*roles]] for i in range(8)]
    assert selection['proposals'].tolist() == proposal
    ref_old, ref_new = explicit(dc,dh,ids,mids,proposal)
    assert torch.allclose(old,ref_old,atol=1e-15,rtol=0) and torch.allclose(candidate,ref_new,atol=1e-15,rtol=0)
    # Ties among current positions select the first in the implementation.
    # Scalar reference is used for ties; exact legacy tie preservation tested separately.
    if not mode.startswith('tie'):
        actual = torch.autograd.grad(candidate,(dc,dh),retain_graph=True)
        expect = torch.autograd.grad(ref_new,(dc,dh),retain_graph=True,allow_unused=True)
        grad_error = max(float((a-b).abs().max()) for a,b in zip(actual,expect) if a.numel())
        assert grad_error < 1e-14
    else:
        grad_error = None
    if mode in ('same','tie'):
        g0 = torch.autograd.grad(old,(dc,dh),retain_graph=True)
        g1 = torch.autograd.grad(candidate,(dc,dh),retain_graph=True)
        assert torch.equal(old,candidate) and all(torch.equal(a,b) for a,b in zip(g0,g1))
    rg = torch.autograd.grad(candidate,roles[:1] if mode == 'duplicate' else roles,retain_graph=True,allow_unused=True)
    assert all(x is None for x in rg)
    gd = torch.cat(torch.autograd.grad(candidate,(dc,dh),retain_graph=True),1)
    active = selection['extra_active']
    assert torch.count_nonzero(gd[active]) == int(active.sum())
    assert candidate <= old
    cases.append({'case':label,'pass':True,'scalar_error':max(abs(float(old-ref_old)),abs(float(candidate-ref_new))),
                  'distance_gradient_max_error':grad_error,'mined_role_gradients_absent':True,
                  'active_extra_pairs':int(active.sum()),'tie_gradient_limit':'Original tie-gradient tested for no extras; scalar convention tested with extras.' if mode.startswith('tie') else None})

# Shared parameters with two separate historical groups. Test chain-rule sum and repeated positions across anchors.
ids=[0,0,1,1,2,2,3,3]
mids=[0,1,2,3,4,5]
weight=torch.randn(10,12,dtype=torch.float64,requires_grad=True)
current_raw=torch.randn(8,10,dtype=torch.float64)
historical_raw=[torch.randn(3,10,dtype=torch.float64),torch.randn(3,10,dtype=torch.float64)]
cur=F.normalize(current_raw@weight,dim=1)
hist_graphs=[F.normalize(v@weight,dim=1) for v in historical_raw]
hist=torch.cat(hist_graphs,0)
dc,dh=torch.cdist(cur,cur),torch.cdist(cur,hist)
roles=[torch.rand(8,14,dtype=torch.float64) for _ in range(3)]
_,direct_loss,sel=module.relation_objectives(dc,dh,ids,mids,roles)
direct=torch.autograd.grad(direct_loss,weight,retain_graph=True)[0]
leaf=hist.detach().requires_grad_()
_,partial_loss,_=module.relation_objectives(torch.cdist(cur,cur),torch.cdist(cur,leaf),ids,mids,roles)
partial,upstream=torch.autograd.grad(partial_loss,(weight,leaf),retain_graph=True)
reconstructed=partial.clone()
for group,coefficient in zip(hist_graphs,upstream.split(3)):
    reconstructed += torch.autograd.grad(group,weight,grad_outputs=coefficient.detach(),retain_graph=True)[0]
error=float((direct-reconstructed).norm()/direct.norm())
assert error < 1e-12 and float(upstream.norm())>0

spec=json.loads((ROOT/'configs/MSVR310/Role-set-gradient-check-v1.json').read_bytes())
math_bytes=Path(spec['math_receipt']).read_bytes()
assert hashlib.sha256(math_bytes).hexdigest()==spec['math_sha256']
math_receipt=json.loads(math_bytes)
assert math_receipt['status']=='PASS_ROLE_SET_MATHEMATICS'
storage=[]
for p in (ROOT.parent,Path('/root/trifusion-storage')):
    usage=shutil.disk_usage(p)
    storage.append({'path':str(p),'device':p.stat().st_dev,'free':usage.free,'total':usage.total})
print(json.dumps({'status':'PASS_INDEPENDENT_CPU_SYNTHETIC_GRADIENT_CHECKS','checked_at':datetime.now(timezone.utc).isoformat(),
                  'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':cases,
                  'two_group_shared_parameter_vjp_relative_error':error,'historical_upstream_norm':float(upstream.norm()),
                  'original_math_receipt_sha256':hashlib.sha256(math_bytes).hexdigest(),'original_math_receipt':math_receipt,
                  'new_real_model_forwards':0,'new_optimizer_updates':0,'new_image_bytes_read':0,'new_remote_artifact_writes':0,
                  'parameter_gradient_witnesses_recomputed':False,'resource_observation_now':storage},indent=2))
