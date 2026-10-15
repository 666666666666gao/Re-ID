"""CPU mathematical/gradient checks; no model or retrieval-performance verdict."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from tools.f1_legal_monotone_ap import (
    relations,monotone_from_scores,smooth_from_scores,cross_environment_ap_from_distances,
)
from tools.msvr_cross_scene_smooth_ap import cross_scene_ap_from_distances


def actual_ap(score,positive,negative):
    values=[]
    for i in range(score.shape[0]):
        n=int(positive[i].sum())
        if n==0:continue
        candidates=torch.where(positive[i]|negative[i])[0].tolist()
        ordered=sorted(candidates,key=lambda j:(-float(score[i,j]),j))
        hits=0;total=0.
        for rank,j in enumerate(ordered,1):
            if positive[i,j]:hits+=1;total+=hits/rank
        values.append(total/n)
    return sum(values)/len(values)


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    ids=[0,0,1,1,2,2];env=[0,1,0,1,0,0]
    mids=[2,3,0,1];menv=[1,0,1,0]
    current=F.normalize(torch.randn(6,7,dtype=torch.float64),dim=1)
    history=F.normalize(torch.randn(4,7,dtype=torch.float64),dim=1)
    d=torch.cdist(current,current);h=torch.cdist(current,history)
    scores=1-torch.cat((d,h),dim=1).square()/2
    positive,negative=relations(scores,ids,mids,env,menv)
    expected=cross_scene_ap_from_distances(d,h,ids,mids,env,menv)
    reproduced=smooth_from_scores(scores,positive,negative)
    assert all(torch.equal(a,b) for a,b in zip(expected,reproduced,strict=True))
    checks=['existing_SmoothAP_score_formula_bitwise_parity']
    for trial in range(100):
        score=(torch.randn(6,10,dtype=torch.float64) if trial else torch.zeros(6,10,dtype=torch.float64)).requires_grad_(True)
        loss,ap,counts=monotone_from_scores(score,positive,negative)
        assert torch.isfinite(loss) and 0<=float(loss.detach())<=1
        assert float(loss.detach())>=1-actual_ap(score.detach(),positive,negative)-1e-12
        gradient,=torch.autograd.grad(loss,score)
        assert bool((gradient[positive]<=1e-12).all())
        assert bool((gradient[negative]>=-1e-12).all())
        assert bool((gradient[~(positive|negative)]==0).all())
    checks+=['100_random_or_exact_tie_AP_upper_envelopes','positive_and_negative_score_monotonicity','ignored_same_environment_same_ID_zero_derivative']
    score=(1-d.square()/2).requires_grad_(True)
    p,n=relations(score,ids,[],env,[])
    loss,ap,counts=monotone_from_scores(score,p,n)
    assert counts.tolist()==[1,1,1,1,0,0]
    assert torch.equal(loss,1-ap[counts>0].mean())
    same_env=[0]*6;p,n=relations(score,ids,[],same_env,[])
    loss,ap,counts=monotone_from_scores(score,p,n)
    gradient,=torch.autograd.grad(loss,score)
    assert float(loss.detach())==0 and int(counts.sum())==0 and bool((gradient==0).all())
    checks+=['variable_legal_positive_count_equal_eligible_query_weight','actual_no_support_graph_connected_zero']
    hard=torch.tensor([[0.,0.,.8]],dtype=torch.float64,requires_grad=True)
    p=torch.tensor([[False,True,False]]);n=torch.tensor([[False,False,True]])
    loss=monotone_from_scores(hard,p,n)[0];gradient,=torch.autograd.grad(loss,hard)
    assert gradient[0,1]<0 and gradient[0,2]>0 and torch.isfinite(gradient).all()
    checks+=['large_wrong_negative_retains_repair_score_derivative']
    weight=torch.randn(9,7,dtype=torch.float64,requires_grad=True)
    a=torch.randn(6,9,dtype=torch.float64);b=torch.randn(4,9,dtype=torch.float64)
    current=F.normalize(a@weight,dim=1);history=F.normalize(b@weight,dim=1)
    current_distance=torch.cdist(current,current);history_distance=torch.cdist(current,history)
    full=cross_environment_ap_from_distances(current_distance,history_distance,ids,mids,env,menv)[0]
    full_grad,=torch.autograd.grad(full,weight,retain_graph=True)
    current_only=cross_environment_ap_from_distances(current_distance,torch.cdist(current,history.detach()),ids,mids,env,menv)[0]
    current_grad,=torch.autograd.grad(current_only,weight,retain_graph=True)
    leaf=history.detach().requires_grad_(True)
    partial=cross_environment_ap_from_distances(current_distance.detach(),torch.cdist(current.detach(),leaf),ids,mids,env,menv)[0]
    upstream,=torch.autograd.grad(partial,leaf)
    replay_grad,=torch.autograd.grad((history*upstream).sum(),weight)
    assert torch.equal(full,current_only) and torch.equal(full,partial)
    assert torch.allclose(full_grad,current_grad+replay_grad,atol=1e-10,rtol=1e-10)
    checks+=['current_and_history_leaf_scalar_bitwise_equal','candidate_side_VJP_matches_full_parameter_gradient']
    print(json.dumps(dict(status='CPU_MATHEMATICAL_AND_GRADIENT_CHECKS_PASS',checks=checks,
       full_vs_replayed_gradient_max_abs=float((full_grad-current_grad-replay_grad).abs().max()),
       scope='Random/hand score and linear-feature math only; NN calls0, no dataset training/performance conclusion, not an originality certificate.')))


if __name__=='__main__':main()
