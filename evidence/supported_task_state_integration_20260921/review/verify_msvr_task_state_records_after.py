"""Replay support clocks and saved moments; not regenerated training gradients."""
from pathlib import Path
import math
import torch
from tools.train_msvr310_signal_oof import sha256


def verify_states(audits,tr,endpoint,warmup):
    observed=0;previous=None;references=0
    for i,audit in enumerate(audits):
        ids=audit['identities'];scenes=audit['scenes']
        candidates=list(zip(ids,scenes))+[(r['identity'],r['scene']) for r in audit['memory']]
        counts=[sum(y==identity and other_scene!=scene for y,other_scene in candidates)
                for identity,scene in zip(ids,scenes,strict=True)]
        assert counts==audit['relation_objective']['cross_scene_positive_counts']
        positive_ids={identity for identity,n in zip(ids,counts,strict=True) if n>0}
        relations={(identity,scene,other_scene) for identity,scene,n in zip(ids,scenes,counts,strict=True)
                   if n>0 for y,other_scene in candidates if y==identity and other_scene!=scene}
        expected=dict(eligible_anchors=sum(n>0 for n in counts),eligible_identities=len(positive_ids),
                      identity_directed_scene_relations=len(relations))
        assert audit['support']==expected
        wanted=i<warmup or expected['eligible_anchors']>0
        assert audit['rank_observed']==wanted
        observed+=int(wanted)
        assert audit['direct_ra_assembly'] and audit['classification_head_gradients_unchanged']
        assert audit['current_rank_backward_calls']==audit['current_auxiliary_backward_calls']==1
        has_reference=bool(audit['direct_single_group_check'])
        assert bool(audit['rank_auxiliary_reference_checks'])==has_reference
        assert audit['direct_component_backward_calls']==(4 if has_reference else 0)
        assert set(audit['actual_parameter_updates'])=={'cnn','transformer','mamba'}
        if has_reference:
            assert set(audit['rank_auxiliary_reference_checks'])=={'cnn','transformer','mamba'}
        states=audit['task_states']
        assert set(states)=={'cnn','transformer','mamba'}
        for role,tasks in states.items():
            from tools.verify_msvr_supported_gradient_balance_stats import pair
            pair(audit['actual_parameter_updates'][role])
            keys={'shared'} if endpoint=='control' else {'auxiliary','rank'} if observed else {'auxiliary'}
            assert set(tasks)==keys
            for key,task in tasks.items():
                assert task['step']==(observed if key=='rank' else i+1)
                assert all(math.isfinite(task[k]) and task[k]>=0 for k in ('first_moment_norm','second_moment_norm'))
            if endpoint=='split' and not wanted and previous and 'rank' in previous[role]:
                assert tasks['rank']==previous[role]['rank']
            if audit['rank_auxiliary_reference_checks']:
                checks=audit['rank_auxiliary_reference_checks'][role]
                assert set(checks)=={'current_rank','historical_rank','full_rank','auxiliary','assembled'}
                for check in checks.values():
                    assert check['passed']
                    error=check['difference_norm']/check['first_norm'] if check['first_norm'] else None
                    assert error==check['relative_l2_error']
                    assert error<=.005 if error is not None else check['difference_norm']<=1e-8
                    references+=1
        previous=states
    assert references==(15 if tr['mode']=='capacity' else 0)
    assert tr['current_rank_backward_calls']==tr['current_auxiliary_backward_calls']==len(audits)
    assert tr['direct_component_backward_calls']==(4 if tr['mode']=='capacity' else 0)
    proof=tr['optimizer_state'];path=Path(proof['path'])
    assert path.is_file() and sha256(path)==proof['sha256'] and path.stat().st_size==proof['bytes']
    assert proof['disk_payload_exact'] and proof['reloaded_moments_exact'] and proof['scaler_exact']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    names=saved['parameter_names'];assert len(names)==len(set(names))==proof['parameter_count']==203
    assert saved['endpoint']==endpoint and tr['task_state_split']==(endpoint=='split')
    state=saved['optimizer']['state'];groups=saved['optimizer']['param_groups']
    assert len(groups)==1 and groups[0]['params']==list(range(203)) and set(state)==set(range(203))
    role_rows={role:[] for role in previous};moment_tensors=0
    for index,name in enumerate(names):
        matched=[role for role in role_rows if name.startswith('encoder.'+role+'_')]
        keys=set(state[index])
        if matched:
            assert len(matched)==1;role=matched[0];role_rows[role].append(index)
            assert keys==set(previous[role])
        else:assert keys=={'head'}
        for key,task in state[index].items():
            assert task['step']==(observed if key=='rank' else len(audits))
            assert task['exp_avg'].dtype==task['exp_avg_sq'].dtype==torch.float32
            assert task['exp_avg'].shape==task['exp_avg_sq'].shape
            assert torch.isfinite(task['exp_avg']).all() and torch.isfinite(task['exp_avg_sq']).all()
            assert (task['exp_avg_sq']>=0).all();moment_tensors+=2
    assert [len(role_rows[r]) for r in ('cnn','transformer','mamba')]==[42,54,93]
    for role,indexes in role_rows.items():
        for key,task in previous[role].items():
            for field,moment in [('first_moment_norm','exp_avg'),('second_moment_norm','exp_avg_sq')]:
                norm=math.sqrt(sum(float(state[i][key][moment].double().square().sum()) for i in indexes))
                assert abs(norm-task[field])<=1e-9*max(1,norm)
    assert moment_tensors==proof['moment_tensors_checked']
    assert saved['scaler']['scale']==tr['steps'][-1]['amp_scale_after']
    return dict(observed_rank_steps=observed,absent_rank_steps=len(audits)-observed,
                moment_tensors=moment_tensors,direct_component_checks=references,
                optimizer_state_sha256=proof['sha256'],scope='Saved state and counters only; parameter-gradient trajectory not reconstructed.')
