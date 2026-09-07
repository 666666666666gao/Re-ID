from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','source'],required=True);a=p.parse_args()
root=Path('C:/Users/gb/.codex_tmp/msvr_freshness_'+a.mode+'_complete_20260907')
r=root/a.mode
s=json.loads((r/'summary.json').read_bytes());cpu=json.loads((r/'cpu_verification.json').read_bytes())
assert s['status']=='COMPLETE_SOURCE_FRESHNESS_MEASUREMENT' and s['mode']==a.mode
assert cpu['status']=='PASS_COMPLETE_FRESHNESS_'+a.mode.upper()
assert cpu['summary_sha256']==hashlib.sha256((r/'summary.json').read_bytes()).hexdigest()
assert s['config_sha256']=='f3a063499b125ef7e90e1b304324a40c967b8d61ba940a6e3193dab5919edae4'
assert s['fresh_loss_updates']==s['heldout_image_reads']==s['official_reads']==0
expected=12 if a.mode=='preflight' else 260
assert len(cpu['all_rows'])==6*expected
results=[];total=0;extra=0;pair={}

def quantiles(v):
    values=np.asarray(v,dtype=np.float64)
    assert np.isfinite(values).all()
    return dict(count=len(values),mean=float(values.mean()),minimum=float(values.min()),
                p05=float(np.quantile(values,.05)),median=float(np.median(values)),
                p95=float(np.quantile(values,.95)),maximum=float(values.max())) if len(values) else dict(count=0)

for e in s['endpoints']:
    path=r/f"fold_{e['fold']}_{e['endpoint']}"
    tr=json.loads((path/'training.json').read_bytes());assert tr==e['training']
    audits=[json.loads(x) for x in (path/'audit.jsonl').read_text().splitlines()]
    rows=[x for x in cpu['all_rows'] if x['fold']==e['fold'] and x['endpoint']==e['endpoint']]
    assert len(audits)==len(rows)==tr['steps']==expected
    assert hashlib.sha256((path/'audit.jsonl').read_bytes()).hexdigest()==tr['audit_sha256']
    assert tr['all_trainable_gradients'] and tr['trainable_tensor_count']==203 and tr['overflow_events']==0
    assert tr['zero_age_reencoding_bitwise'] and e['strict_state_reload']
    pairs=[];memory_rows=[];ages=Counter()
    for i,(audit,row) in enumerate(zip(audits,rows,strict=True),1):
        assert audit['step']==row['step']==i and audit['epoch']==row['epoch']
        assert row['memory_records']==len(audit['memory'])==len(row['drift'])
        assert row['drift']==audit['cache_l2_drift'] and row['parameter_gradients']==audit['parameter_gradients']
        assert audit['fresh_role_record_forwards']==64*len(set(x['stored_step'] for x in audit['memory']))
        assert not audit['refreshed_loss_used_for_update'] and audit['amp_scale_after']>=audit['amp_scale_before']
        assert abs(row['stale_loss']-audit['stale_statistics']['expanded_triplet'])<2e-6
        assert abs(row['fresh_loss']-audit['fresh_statistics']['expanded_triplet'])<2e-6
        if row['memory_records']:
            ages.update(row['historical_ages']);memory_rows.append(row)
        pairs.append((audit['record_indices'],audit['pixel_sha256']))
    pair[(e['fold'],e['endpoint'])]=pairs
    assert tr['extra_role_record_forwards']==64+sum(x['fresh_role_record_forwards'] for x in audits)
    total+=len(audits);extra+=tr['extra_role_record_forwards']
    role={}
    for name in ['cnn','transformer','mamba']:
        g=[x['parameter_gradients'][name] for x in memory_rows]
        role[name]=dict(fresh_cosine=quantiles([x['fresh']['cosine'] for x in g if x['fresh']['cosine'] is not None]),
                        duplicate_cosine=quantiles([x['duplicate_noise']['cosine'] for x in g if x['duplicate_noise']['cosine'] is not None]),
                        fresh_difference_norm=quantiles([x['fresh']['difference_norm'] for x in g]),
                        duplicate_difference_norm=quantiles([x['duplicate_noise']['difference_norm'] for x in g]),
                        steps_difference_exceeds_duplicate=sum(x['fresh']['difference_norm']>x['duplicate_noise']['difference_norm'] for x in g),
                        zero_or_undefined_cosine=sum(x['fresh']['cosine'] is None for x in g))
    pairs_count=sum(64*x['memory_records'] for x in memory_rows)
    result=dict(fold=e['fold'],endpoint=e['endpoint'],steps=len(audits),history_steps=len(memory_rows),
                candidate_exposures=sum(x['memory_records'] for x in memory_rows),age_exposures=dict(sorted(ages.items())),
                fresh_minus_stale_loss=quantiles([x['fresh_loss']-x['stale_loss'] for x in memory_rows]),
                cache_l2_drift=quantiles([d for x in memory_rows for d in x['drift']]),
                distance_error_pair_weighted_mean=sum(x['distance_error_mean']*64*x['memory_records'] for x in memory_rows)/pairs_count,
                distance_error_max=max(x['distance_error_max'] for x in memory_rows),
                positive_winner_changes=sum(x['positive_winner_changes'] for x in memory_rows),
                negative_winner_changes=sum(x['negative_winner_changes'] for x in memory_rows),
                stale_wrong_fresh_correct=sum(x['stale_wrong_fresh_correct'] for x in memory_rows),
                stale_correct_fresh_wrong=sum(x['stale_correct_fresh_wrong'] for x in memory_rows),
                roles=role,extra_role_record_forwards=tr['extra_role_record_forwards'],
                history=tr['history'],peak_allocated_mib=tr['peak_allocated_mib'])
    results.append(result)
for f in range(3): assert pair[(f,'control')]==pair[(f,'instance_memory')]
report=dict(status='PASS_COMPLETE_FRESHNESS_TEXT_REAGGREGATION',mode=a.mode,total_steps=total,
            distance_elements=cpu['distance_elements'],all_endpoints=results,extra_role_record_forwards=extra,
            original_q1_unchanged=True,official_reads=0,whole_goal_achieved=False,
            limitations='Reaggregates all registered text rows. Runtime actual parameter-gradient witnesses are not independently recomputed here. Preflight uses a distinct short capacity trajectory; it cannot establish full-training or heldout effects.')
(root/'local_complete_reaggregation.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
print(json.dumps(dict(status=report['status'],mode=a.mode,total_steps=total,distance_elements=cpu['distance_elements'],extra_role_record_forwards=extra,
                     endpoints=[dict(fold=x['fold'],endpoint=x['endpoint'],history_steps=x['history_steps'],
                                     distance_error_mean=x['distance_error_pair_weighted_mean'],
                                     fresh_minus_stale_loss_mean=x['fresh_minus_stale_loss']['mean'],
                                     role_mean_cosines={k:v['fresh_cosine'].get('mean') for k,v in x['roles'].items()}) for x in results])))
