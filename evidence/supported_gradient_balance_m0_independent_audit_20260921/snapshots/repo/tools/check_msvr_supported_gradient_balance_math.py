"""Synthetic CPU differentiation checks; not model or retrieval evidence."""
import copy
import math

import torch

from tools.msvr_supported_gradient_balance import (
    ROLES, SupportedBalance, role_indices, scalar_gradients, combine, support_counts,
)


def checks():
    names=['encoder.'+e+'_fixture.weight' for e in ROLES]
    groups=role_indices(names)
    snapshots=[]
    for scale in (1.,256.):
        parameters=[torch.nn.Parameter(torch.tensor([.3+i*.1,-.4+i*.03])) for i in range(3)]
        head=torch.nn.Parameter(torch.tensor([.2]))
        x=torch.tensor([.7,-1.2]);y=torch.tensor([-.1,.9]);target=torch.tensor([.8,-.2])
        u=[p*x for p in parameters];v=[p*y for p in parameters]
        rank=sum((a-b).square().sum() for a,b in zip(u,v,strict=True))
        rank_current=sum((a-b.detach()).square().sum() for a,b in zip(u,v,strict=True))
        rank_history=sum((a.detach()-b).square().sum() for a,b in zip(u,v,strict=True))
        auxiliary=sum((a-target).square().sum() for a in u)+head.square().sum()
        r=scalar_gradients(rank_current,parameters,scale)
        h=scalar_gradients(rank_history,parameters,scale)
        current=scalar_gradients(rank_current+auxiliary,parameters,scale)
        direct_rank=scalar_gradients(rank,parameters,scale)
        direct_aux=scalar_gradients(auxiliary,parameters,scale)
        direct_total=scalar_gradients(rank+auxiliary,parameters,scale)
        head.grad=torch.autograd.grad((rank_current+auxiliary)*scale,head,retain_graph=True)[0]
        head_before=head.grad.clone()
        for p,g in zip(parameters,current,strict=True):p.grad=g.clone()
        controller=SupportedBalance()
        control,full,a=combine(parameters,current,r,direct_aux,h,scale,groups,controller,True,False)
        assert all(torch.allclose(g,f,rtol=1e-6,atol=1e-6*scale) for g,f in zip(full,direct_rank,strict=True))
        assert all(torch.allclose(g,f,rtol=1e-6,atol=1e-6*scale) for g,f in zip(a,direct_aux,strict=True))
        assert all(torch.allclose(p.grad,f,rtol=1e-6,atol=1e-6*scale) for p,f in zip(parameters,direct_total,strict=True))
        controller=SupportedBalance()
        rows,_,_=combine(parameters,current,r,direct_aux,h,scale,groups,controller,True,True)
        for role,indexes in groups.items():
            wr=rows[role]['applied_rank_weight'];wa=rows[role]['applied_auxiliary_weight']
            for i in indexes:
                expected=wr*direct_rank[i]+wa*direct_aux[i]
                assert torch.allclose(parameters[i].grad,expected,rtol=1e-6,atol=1e-6*scale)
        assert torch.equal(head.grad,head_before)
        snapshots.append(copy.deepcopy(controller.states))
        before=copy.deepcopy(controller.states)
        zero=[torch.zeros_like(p) for p in parameters]
        combine(parameters,current,zero,current,zero,scale,groups,controller,False,True)
        assert controller.states==before
        assert all(torch.equal(p.grad,g) for p,g in zip(parameters,current,strict=True))
        assert torch.equal(head.grad,head_before)
    assert snapshots[0]==snapshots[1]

    c=SupportedBalance()
    for role in ROLES:
        no=c.observe(role,0.,2.,False)
        assert no['before'] is no['after'] is None and no['proposed_rank_weight']==1
        first=c.observe(role,4.,16.,True)
        assert first['after']==dict(rank=4.,auxiliary=16.,supported_steps=1)
        assert abs(first['ratio']-math.sqrt((16+1e-12)/(4+1e-12)))<1e-15
        c.observe(role,0.,999.,False)
        second=c.observe(role,8.,8.,True)
        assert second['after']==dict(rank=.9*4+.1*8,auxiliary=.9*16+.1*8,supported_steps=2)
    assert c.states['cnn']==c.states['transformer']==c.states['mamba']
    for r,a,wanted in ((0.,1.,4.),(1.,0.,.25),(0.,0.,1.)):
        result=SupportedBalance().observe('cnn',r,a,True)
        assert result['ratio']==wanted
        assert .4<=result['proposed_rank_weight']<=1.6
        assert .4<=result['proposed_auxiliary_weight']<=1.6
        assert result['proposed_rank_weight']+result['proposed_auxiliary_weight']==2.
    assert role_indices(list(reversed(names)))==dict(cnn=[2],transformer=[1],mamba=[0])
    # Multiple tensors per role and a real permutation of the same update.
    block_names=['encoder.'+e+'_'+part for e in ROLES for part in ('weight','bias')]
    values=[torch.tensor([i+.5,1.-i*.2]) for i in range(6)]
    block_current=[v*3 for v in values]
    block_rank=[v*.5 for v in values]
    block_history=[v*.25 for v in values]
    block_auxiliary=[v*2.5+torch.tensor([.02,-.01]) for v in values]
    def block_run(order,apply=True,supported=True):
        ps=[torch.nn.Parameter(values[i].clone()) for i in order]
        for p in ps:p.grad=torch.zeros_like(p)
        controller=SupportedBalance()
        rows,_,_=combine(ps,[block_current[i] for i in order],
                         [block_rank[i] if supported else torch.zeros_like(block_rank[i]) for i in order],
                         [block_auxiliary[i] for i in order],
                         [block_history[i] if supported else torch.zeros_like(block_history[i]) for i in order],
                         1.,role_indices([block_names[i] for i in order]),controller,supported,apply)
        return {i:p.grad.clone() for i,p in zip(order,ps,strict=True)},rows,controller.states
    original,original_rows,original_state=block_run(list(range(6)))
    permuted,permuted_rows,permuted_state=block_run([4,5,0,1,2,3])
    assert original_state==permuted_state and original_rows==permuted_rows
    assert all(torch.equal(original[i],permuted[i]) for i in original)
    for role,indexes in role_indices(block_names).items():
        expected_rank=math.sqrt(sum(float((block_rank[i]+block_history[i]).double().square().sum()) for i in indexes))
        expected_aux=math.sqrt(sum(float(block_auxiliary[i].double().square().sum()) for i in indexes))
        assert abs(original_state[role]['rank']-expected_rank)<1e-12
        assert abs(original_state[role]['auxiliary']-expected_aux)<1e-12
        assert original_rows[role]['subtraction_auxiliary_vs_direct']['difference_norm']>0
        assert original_rows[role]['direct_sum_vs_original']['difference_norm']>0
    # Exercise the real CPU scalar verifier when direct and combined gradients differ.
    from tools.probe_msvr_history_candidate_gradients import compare
    from tools.verify_msvr_supported_gradient_balance_stats import verify_balance
    for endpoint,supported in (('control',True),('balanced',True),('balanced',False)):
        grads,rows,states=block_run(list(range(6)),endpoint=='balanced',supported)
        if endpoint=='control' or not supported:
            assert all(torch.equal(grads[i],block_current[i]+(block_history[i] if supported else 0)) for i in grads)
        ids=[0,0] if supported else [0,1]
        scenes=[0,1];counts=[1,1] if supported else [0,0]
        audit=dict(relation_objective=dict(cross_scene_positive_counts=counts),identities=ids,scenes=scenes,
            memory=[],support=support_counts(ids,scenes,[],counts),classification_head_gradients_unchanged=True,
            current_rank_backward_calls=1,current_auxiliary_backward_calls=1,gradient_balance=rows,
            actual_parameter_updates={e:compare([values[i] for i in indexes],[values[i] for i in indexes])
                                      for e,indexes in role_indices(block_names).items()},
            applied_gradients={e:rows[e]['rank_vs_applied'] for e in ROLES},
            rank_auxiliary_reference_checks={},direct_component_backward_calls=0,direct_single_group_check={})
        training=dict(gradient_balance_state=states,current_rank_backward_calls=1,current_auxiliary_backward_calls=1,
            direct_component_backward_calls=0,gradient_balancing_applied=endpoint=='balanced',
            classification_head_rule='original_current_total_gradient')
        proof=verify_balance([audit],training,endpoint,0)
        assert proof['supported_steps']==int(supported)
    supported=support_counts([0,0,1],[0,1,0],[dict(identity=1,scene=1)],[1,1,1])
    assert supported==dict(eligible_anchors=3,eligible_identities=2,identity_directed_scene_relations=3)
    assert not torch.cuda.is_initialized()
    return dict(status='PASS_SYNTHETIC_SUPPORTED_GRADIENT_BALANCE',model_forwards=0,optimizer_updates=0,
                current_history_chain_rule=True,direct_auxiliary=True,weighted_direct_reference=True,
                amp_power_of_two_invariance=True,unsupported_state_unchanged=True,
                unsupported_update_unchanged=True,classification_head_unchanged=True,
                bounded_coefficients=True,whole_role_grouping=True,explicit_zero_norm_definition=True,
                multi_tensor_role_norm=True,role_permutation_equivariance=True,
                direct_auxiliary_distinguished_from_subtraction=True,
                scalar_verifier_combined_control_direct_candidate_and_zero_support=True)
