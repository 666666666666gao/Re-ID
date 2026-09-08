from datetime import datetime
import json,math,time
import numpy as np
import torch

# MODULE_SOURCE is prepended by the local builder and its SHA is recorded.
torch.set_num_threads(2)
started=time.monotonic()
ids=[0,0,1,1];mids=[0,1]
d=torch.tensor([[0,.94,.99,1.02,.98,1.04],[.94,0,1.03,.97,1.01,.99],[.99,1.03,0,.96,1.04,.98],[1.02,.97,.96,0,.99,1.01]],dtype=torch.float64)

def reference(values,identities,candidates):
    result=[]
    for i,row in enumerate(values):
        pos=[j for j,y in enumerate(candidates) if y==identities[i] and j!=i]
        allpos=[j for j in range(len(candidates)) if j!=i]
        scores=1-row*row/2
        terms=[]
        for p in pos:
            rp=1+sum(1/(1+math.exp(-(float(scores[j]-scores[p]))/.01)) for j in pos if j!=p)
            ra=1+sum(1/(1+math.exp(-(float(scores[j]-scores[p]))/.01)) for j in allpos if j!=p)
            terms.append(rp/ra)
        result.append(sum(terms)/len(terms))
    return np.array(result)

errors=[]
for history_size in (0,2):
    x=d[:,:4].clone().requires_grad_();y=d[:,4:4+history_size].clone().requires_grad_()
    hids=mids[:history_size]
    loss,ap=smooth_ap_from_distances(x,y,ids,hids)
    ref=reference(d[:,:4+history_size].numpy(),ids,ids+hids)
    error=float(np.max(np.abs(ap.detach().numpy()-ref)));errors.append(error);assert error<1e-12
    assert torch.autograd.gradcheck(lambda a,b:smooth_ap_from_distances(a,b,ids,hids)[0],(x,y),eps=1e-6,atol=1e-6,rtol=1e-4)
    gx,gy=torch.autograd.grad(loss,(x,y));assert torch.isfinite(gx).all() and torch.isfinite(gy).all()
    assert torch.count_nonzero(gx.diag())==0
    perm=torch.tensor([2,3,0,1]);hp=torch.tensor(list(reversed(range(history_size))))
    if history_size==0:hp=hp.long()
    xx=x.detach()[perm][:,perm];yy=y.detach()[perm][:,hp]
    ll,_=smooth_ap_from_distances(xx,yy,[ids[i] for i in perm],[hids[i] for i in hp])
    assert torch.allclose(loss,ll,atol=1e-12,rtol=0)
# Exact positive/negative tie: sigmoid comparison is .5, not strict evaluation tie-breaking.
tie=torch.ones(4,4,dtype=torch.float64);tie.fill_diagonal_(0)
_,ap=smooth_ap_from_distances(tie,torch.empty(4,0,dtype=torch.float64),ids,[])
assert torch.allclose(ap,torch.full((4,),.5,dtype=torch.float64),atol=1e-12,rtol=0)
# Real API-sized synthetic coverage includes legitimate class zero and repeated identity positions.
gen=torch.Generator().manual_seed(42)
z=torch.nn.functional.normalize(torch.randn(128,32,generator=gen),dim=1)
dc=torch.cdist(z[:64],z[:64]);dh=torch.cdist(z[:64],z[64:])
loss,ap=smooth_ap_from_distances(dc,dh,[i//8 for i in range(64)],[i//8 for i in range(64)])
assert torch.isfinite(loss) and ((ap>0)&(ap<=1)).all()
print(json.dumps(dict(status='PASS_SYNTHETIC_SMOOTH_AP_MATH',completed_at=datetime.now().astimezone().isoformat(),module_sha256=MODULE_SHA256,maximum_scalar_reference_error=max(errors),finite_difference_checks=2,self_distance_gradient_zero=True,permutation_checks=2,tie_ap=.5,synthetic_large_shape=[64,128],elapsed_seconds=time.monotonic()-started,model_forwards=0,optimizer_updates=0,dataset_reads=0,scope='Synthetic formula/derivative tests only; no actual TriFusion parameter gradient or retrieval result.')))
