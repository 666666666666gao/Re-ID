"""CPU mathematical checks; these do not establish real model gradients."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.msvr_role_set_relations import relation_objectives, fused_distances
from tools.probe_msvr_history_candidate_gradients import differentiable_history_loss


def checks():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    ids = [0,0,1,1,2,2,3,3]
    mids = [0,1,4,5]
    current = torch.randn(8,16,requires_grad=True)
    raw = torch.randn(4,16)
    matrix = torch.randn(16,16,requires_grad=True)
    history = F.normalize(raw@matrix,dim=1)
    dc,dh = fused_distances(current,history)
    same = torch.cat((dc,dh),dim=1).detach()
    old = differentiable_history_loss(current,ids,history,[dict(identity=i) for i in mids])
    hard,candidate,selection = relation_objectives(dc,dh,ids,mids,[same]*3)
    assert torch.equal(old,hard) and torch.equal(hard,candidate)
    for a,b in zip(torch.autograd.grad(old,(current,matrix),retain_graph=True),
                   torch.autograd.grad(candidate,(current,matrix),retain_graph=True),strict=True):
        assert torch.equal(a,b)
    assert selection['counts'].eq(1).all() and not selection['extra'].any()
    # Force different legal candidate proposals without changing fused geometry.
    roles = [torch.arange(12,dtype=torch.float32).repeat(8,1).roll(k,1) for k in (0,4,8)]
    hard,candidate,selection = relation_objectives(dc,dh,ids,mids,roles)
    assert (selection['counts']>1).all() and candidate<=hard
    assert selection['extra_active'].any()
    dg = torch.autograd.grad(candidate,(dc,dh),retain_graph=True)
    joined = torch.cat(dg,dim=1)
    assert (joined[selection['extra_active']] != 0).all()
    direct = torch.autograd.grad(candidate,matrix,retain_graph=True)[0]
    leaf = history.detach().requires_grad_(True)
    lc,lh = fused_distances(current.detach(),leaf)
    _,loss,_ = relation_objectives(lc,lh,ids,mids,roles)
    upstream = torch.autograd.grad(loss,leaf)[0]
    split = torch.autograd.grad(history,matrix,grad_outputs=upstream,retain_graph=True)[0]
    assert torch.equal(direct,split) and direct.abs().sum()>0
    # No-history is the actual prewarmup/fixed-overfit queue condition.
    empty=history[:0]
    ec,eh=fused_distances(current,empty)
    a,b,s=relation_objectives(ec,eh,ids,[],[ec.detach()]*3)
    assert torch.equal(a,b) and s['counts'].eq(1).all()
    # An exact current/history negative tie retains the old hard derivative.
    tc=torch.full((4,4),.6,requires_grad=True)
    th=torch.full((4,2),.6,requires_grad=True)
    tied=torch.cat((tc,th),dim=1).detach()
    a,b,s=relation_objectives(tc,th,[0,0,1,1],[0,1],[tied]*3)
    ga=torch.autograd.grad(a,(tc,th),retain_graph=True)
    gb=torch.autograd.grad(b,(tc,th),retain_graph=True)
    assert all(torch.equal(x,y) for x,y in zip(ga,gb,strict=True))
    # All roles agreeing on the same additional negative adds it once only.
    a,b,s=relation_objectives(dc,dh,ids,mids,[roles[0]]*3)
    assert s['counts'].le(2).all()
    return dict(status='PASS_ROLE_SET_MATHEMATICS',seed=42,legal_class_zero=True,
                original_scalar_and_gradient_preserved_without_extras=True,
                original_tie_gradient_preserved_without_extras=True,
                unique_negative_positions=True,extra_negative_derivatives_nonzero=True,
                current_and_history_gradients=True,history_chain_rule_exact=True,
                no_history_supported=True,model_forwards=0,optimizer_updates=0,
                scope='Synthetic CPU mathematics, not a real-model gradient or generalization result.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=checks()
    assert not args.output.exists()
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
