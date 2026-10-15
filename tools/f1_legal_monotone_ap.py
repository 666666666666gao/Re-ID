"""Legal AP surrogate using standard hard positive ranks and log2 softplus.

The rank construction follows the ROADMAP/SupRank principle, with the standard
logistic step upper envelope. Neither primitive is claimed as new. This adapter
defines the project's camera/scene ignore set and current/history gradient path.
"""
import math
import torch
import torch.nn.functional as F

from tools.msvr_smooth_ap import TAU, paired_objectives


def relations(score, identities, history_identities, environments, history_environments):
    ids=torch.as_tensor(identities,device=score.device)
    mids=torch.as_tensor(history_identities,device=score.device,dtype=ids.dtype)
    env=torch.as_tensor(environments,device=score.device)
    menv=torch.as_tensor(history_environments,device=score.device,dtype=env.dtype)
    same_id=ids[:,None]==torch.cat((ids,mids))[None,:]
    same_env=env[:,None]==torch.cat((env,menv))[None,:]
    assert same_id.shape==score.shape and same_env.shape==score.shape
    positive=same_id&~same_env
    negative=~same_id
    assert negative.any(1).all()
    return positive,negative


def monotone_from_scores(score,positive,negative):
    counts=positive.sum(1);eligible=counts>0
    positions=torch.arange(score.shape[1],device=score.device)
    aps=[]
    for i in range(score.shape[0]):
        pos=positions[positive[i]]
        if pos.numel()==0:
            aps.append(score[i].sum()*0)
            continue
        difference=score[i][None,:]-score[i,pos][:,None]
        # A fixed candidate-index tie order makes the hard positive count exact.
        earlier=(difference.detach()>0)|((difference.detach()==0)&(positions[None,:]<pos[:,None]))
        rank_plus=1+(earlier&positive[i][None,:]).sum(1).to(score.dtype)
        rank_minus=(F.softplus(difference/TAU)/math.log(2)).masked_fill(~negative[i][None,:],0).sum(1)
        aps.append((rank_plus/(rank_plus+rank_minus)).mean())
    per_anchor=torch.stack(aps)
    loss=1-per_anchor[eligible].mean() if bool(eligible.any()) else score.sum()*0
    return loss,per_anchor,counts


def smooth_from_scores(score,positive,negative):
    """Existing Smooth-AP formula, used only for score diagnostics."""
    positions=torch.arange(score.shape[1],device=score.device)
    counts=positive.sum(1);eligible=counts>0;aps=[]
    for i in range(score.shape[0]):
        pos=positions[positive[i]]
        if pos.numel()==0:
            aps.append(score[i].sum()*0)
            continue
        comparison=torch.sigmoid((score[i][None,:]-score[i,pos][:,None])/TAU)
        valid=(positive[i]|negative[i])[None,:]&(positions[None,:]!=pos[:,None])
        comparison=comparison.masked_fill(~valid,0)
        numerator=1+(comparison*positive[i][None,:]).sum(1)
        aps.append((numerator/(1+comparison.sum(1))).mean())
    per_anchor=torch.stack(aps)
    return (1-per_anchor[eligible].mean() if bool(eligible.any()) else score.sum()*0),per_anchor,counts


def cross_environment_ap_from_distances(current,historical,identities,history_identities,environments,history_environments):
    assert current.shape==(len(identities),len(identities)) and historical.shape[0]==len(identities)
    assert current.dtype in (torch.float32,torch.float64) and historical.dtype==current.dtype
    with torch.autocast(current.device.type,enabled=False):
        score=1-torch.cat((current,historical),dim=1).square()/2
        positive,negative=relations(score,identities,history_identities,environments,history_environments)
        return monotone_from_scores(score,positive,negative)


def objectives(current,historical,identities,history_identities,environments,history_environments):
    hard,standard,standard_ap=paired_objectives(current,historical,identities,history_identities)
    cross,cross_ap,counts=cross_environment_ap_from_distances(current,historical,identities,history_identities,environments,history_environments)
    return hard,standard,cross,standard_ap,cross_ap,counts


def score_diagnostics(current,historical,identities,history_identities,environments,history_environments):
    """One recorded eligible update per epoch; score leaves require no NN replay."""
    with torch.autocast(current.device.type,enabled=False):
        score=(1-torch.cat((current,historical),dim=1).detach().square()/2).requires_grad_(True)
        positive,negative=relations(score,identities,history_identities,environments,history_environments)
        old=smooth_from_scores(score,positive,negative)[0]
        new=monotone_from_scores(score,positive,negative)[0]
        old_grad,=torch.autograd.grad(old,score,retain_graph=True)
        new_grad,=torch.autograd.grad(new,score)
        gaps=[]
        for i in range(score.shape[0]):
            gaps.append((score[i,negative[i]][None,:]-score[i,positive[i]][:,None]).reshape(-1))
        gaps=torch.cat(gaps);wrong=gaps[gaps>0]
        return dict(scope='first eligible update per epoch, same pre-update train scores; no parameter gradient or NN replay',
                    old_smooth_ap_loss=float(old.detach()),monotone_ap_loss=float(new.detach()),
                    legal_positive_scores=int(positive.sum()),legal_negative_scores=int(negative.sum()),
                    misordered_pairs=int(wrong.numel()),
                    misorders_above_10tau=int((wrong>10*TAU).sum()),misorders_above_20tau=int((wrong>20*TAU).sum()),
                    old_positive_score_gradient_above_1e_minus12=int((old_grad[positive]>1e-12).sum()),
                    new_positive_score_gradient_above_1e_minus12=int((new_grad[positive]>1e-12).sum()),
                    new_negative_score_gradient_below_minus1e_minus12=int((new_grad[negative]<-1e-12).sum()))
