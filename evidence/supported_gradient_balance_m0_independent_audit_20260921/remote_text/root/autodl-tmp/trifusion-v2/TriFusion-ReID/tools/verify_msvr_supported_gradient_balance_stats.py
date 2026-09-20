"""Replay controller statistics from saved scalars, not parameter gradients."""
import math

ROLES=('cnn','transformer','mamba')


def close(a,b):
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a-b)<=1e-7*max(1,abs(a),abs(b)), (a,b)


def pair(p):
    a,b,d=p['first_norm'],p['second_norm'],p['difference_norm']
    c=p['cosine']
    assert all(math.isfinite(x) and x>=0 for x in (a,b,d))
    assert (c is None)==(a==0 or b==0)
    if c is not None:
        assert abs(c)<=1.00001
        close(d*d,a*a+b*b-2*a*b*c)
    else:
        close(d*d,a*a+b*b)


def verify_balance(audits,training,endpoint,warmup):
    states={e:None for e in ROLES}
    supported_steps=reference_steps=zero_supported_steps=0
    for index,row in enumerate(audits):
        counts=row['relation_objective']['cross_scene_positive_counts']
        ids=row['identities'];scenes=row['scenes']
        candidates=list(zip(ids,scenes))+[(r['identity'],r['scene']) for r in row['memory']]
        positive_ids={identity for identity,n in zip(ids,counts,strict=True) if n>0}
        relations={(identity,scene,other_scene)
                   for identity,scene,n in zip(ids,scenes,counts,strict=True) if n>0
                   for other_id,other_scene in candidates if identity==other_id and scene!=other_scene}
        expected=dict(eligible_anchors=sum(n>0 for n in counts),eligible_identities=len(positive_ids),
                      identity_directed_scene_relations=len(relations))
        assert row['support']==expected
        supported=index>=warmup and expected['eligible_anchors']>0
        supported_steps+=int(supported)
        zero_supported_steps+=int(index>=warmup and not supported)
        assert row['classification_head_gradients_unchanged'] and row['current_rank_backward_calls']==1
        assert row['current_auxiliary_backward_calls']==1
        assert set(row['gradient_balance'])==set(row['actual_parameter_updates'])==set(ROLES)
        for role in ROLES:
            b=row['gradient_balance'][role]
            for key in ('rank_vs_auxiliary','current_rank_vs_history','rank_vs_applied',
                        'auxiliary_vs_applied','original_sum_vs_applied',
                        'subtraction_auxiliary_vs_direct','direct_sum_vs_original'):
                pair(b[key])
            pair(row['actual_parameter_updates'][role])
            p=b['rank_vs_auxiliary'];r,a=p['first_norm'],p['second_norm']
            ra=0 if p['cosine'] is None else r*a*p['cosine']
            assert b['supported']==supported and b['before']==states[role]
            before=states[role]
            ratio=None;wr=wa=1.0
            if supported:
                states[role]=(dict(rank=r,auxiliary=a,supported_steps=1) if before is None else
                    dict(rank=.9*before['rank']+.1*r,auxiliary=.9*before['auxiliary']+.1*a,
                         supported_steps=before['supported_steps']+1))
                ratio=min(4.,max(.25,((states[role]['auxiliary']+1e-12)/(states[role]['rank']+1e-12))**.5))
                wr=2*ratio/(1+ratio);wa=2/(1+ratio)
            assert b['after']==states[role]
            assert b['ratio']==ratio
            close(wr,b['proposed_rank_weight']);close(wa,b['proposed_auxiliary_weight'])
            if endpoint=='control' or not supported:wr=wa=1.
            close(wr,b['applied_rank_weight']);close(wa,b['applied_auxiliary_weight'])
            assert .4<=wr<=1.6 and .4<=wa<=1.6
            close(wr+wa,2.)
            actual=b['rank_vs_applied']['second_norm']
            close(actual,b['auxiliary_vs_applied']['second_norm'])
            close(actual,b['original_sum_vs_applied']['second_norm'])
            close(actual,row['applied_gradients'][role]['second_norm'])
            close(b['subtraction_auxiliary_vs_direct']['second_norm'],a)
            close(b['direct_sum_vs_original']['first_norm']**2,r*r+a*a+2*ra)
            close(b['direct_sum_vs_original']['second_norm'],b['original_sum_vs_applied']['first_norm'])
            if endpoint=='balanced' and supported:
                close(actual**2,wr*wr*r*r+wa*wa*a*a+2*wr*wa*ra)
            else:
                assert b['original_sum_vs_applied']['difference_norm']==0
            history=b['current_rank_vs_history'];u,v=history['first_norm'],history['second_norm']
            uv=0 if history['cosine'] is None else u*v*history['cosine']
            close(r*r,u*u+v*v+2*uv)
            if index>=warmup and not supported:
                assert r==u==v==0
            checks=row['rank_auxiliary_reference_checks']
            if checks:
                assert set(checks)==set(ROLES)
                assert set(checks[role])=={'current_rank','historical_rank','full_rank','auxiliary','applied'}
                for ref in checks[role].values():
                    pair(ref)
                    assert ref['passed']
                    if ref['first_norm']:
                        close(ref['relative_l2_error'],ref['difference_norm']/ref['first_norm'])
                        assert ref['relative_l2_error']<=.005
                    else:
                        assert ref['relative_l2_error'] is None and ref['difference_norm']<=1e-8
        reference_steps+=int(bool(row['rank_auxiliary_reference_checks']))
        assert row['direct_component_backward_calls']==(4 if row['rank_auxiliary_reference_checks'] else 0)
        assert bool(row['rank_auxiliary_reference_checks'])==bool(row['direct_single_group_check'])
    assert training['gradient_balance_state']==states
    assert training['current_rank_backward_calls']==len(audits)
    assert training['current_auxiliary_backward_calls']==len(audits)
    assert training['direct_component_backward_calls']==4*reference_steps
    assert training['gradient_balancing_applied']==(endpoint=='balanced')
    assert training['classification_head_rule']=='original_current_total_gradient'
    return dict(steps=len(audits),supported_steps=supported_steps,zero_supported_steps=zero_supported_steps,
                direct_reference_steps=reference_steps,all_scalar_state_updates_replayed=True,
                parameter_gradient_scope='runtime witnesses; full per-step gradient vectors not regenerated')
