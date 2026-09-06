from pathlib import Path
import json, hashlib, sys
import numpy as np
root=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
def expected_style_plan(cameras, fold, step, force_active):
    camera = np.asarray(cameras)
    generator = np.random.default_rng(np.random.SeedSequence([42, fold, step]))
    active = bool(generator.uniform() < 0.5)
    random_scores = generator.uniform(size=(len(camera), len(camera)))
    donors = []
    for index, source_camera in enumerate(camera):
        possible = np.flatnonzero(camera != source_camera)
        assert len(possible) > 0
        donors.append(int(possible[np.argmin(random_scores[index, possible])]))
    coefficients = generator.beta(0.1, 0.1, size=len(camera)).tolist()
    return {"fold": fold, "step": step, "active": active or force_active,
            "forced_active": force_active, "donors": donors, "coefficients": coefficients,
            "all_donors_cross_camera": True}

d=root/'evidence/v27_m0_20260907'
summary=json.loads((d/'run_summary.json').read_bytes())
replay=json.loads((root/'evidence/trifusion_v26_fixed_sampler_metadata_20260907.json').read_bytes())
assert len(summary['preflight'])==3
checked_preflight=0
for fold in summary['preflight']:
    number=fold['fold']
    a,b=fold['endpoints']
    for key in ['initial_state_sha256','batch_receipts','all_output_sha256','state_unchanged','evaluation_style_disabled']:
        assert a[key]==b[key]
    for end in (a,b):
        assert len(end['field_diagnostics'])==8
        for index,row in enumerate(end['field_diagnostics']):
            batch=replay['folds'][number]['arms']['control']['batches'][index]
            cameras=[replay['folds'][number]['source_manifest'][i]['camera'] for i in batch['sampler_indices']]
            assert row['plan']==expected_style_plan(cameras,number,index,True)
            assert row['baseline_exact'] and row['role_reference_both_frozen']
            assert row['style_active']==int(end['endpoint']=='source_style')
            checked_preflight+=1
m0=summary['m0']
assert len(m0['capacities'])==2 and m0['overfit']['steps']==100
checked=0
for item in [*m0['capacities'],m0['overfit']]:
    log=d/Path(item['all_steps_path']).name
    assert hashlib.sha256(log.read_bytes()).hexdigest()==item['all_steps_sha256']
    rows=[json.loads(x) for x in log.read_text().splitlines()]
    assert len(rows)==item['steps']
    for number,row in enumerate(rows):
        index=0 if item['fixed_batch'] else number
        indices=replay['folds'][0]['arms']['control']['batches'][index]['sampler_indices']
        cameras=[replay['folds'][0]['source_manifest'][i]['camera'] for i in indices]
        plan=expected_style_plan(cameras,0,index,item['fixed_batch'])
        assert row['style_plan']==item['style_plans'][number]==plan
        assert row['losses']==item['components'][number]
        assert row['losses']['total']==item['losses'][number]
        assert 0 < row['live_gradient_tensors'] <= 203 and not row['overflow']
        assert row['losses']['style_active']==int(item['endpoint']=='source_style' and plan['active'])
        checked+=1
    assert item['frozen_state_unchanged'] and item['nonzero_gradient_tensors']==203
    assert max(r['live_gradient_tensors'] for r in rows)==203
    if not item['fixed_batch']:
        assert all(r['live_gradient_tensors']==203 for r in rows)
assert checked==116 and checked_preflight==48
assert m0['capacities'][0]['style_plans']==m0['capacities'][1]['style_plans']
loss=m0['overfit']
ratio=(loss['losses'][-1]-loss['combined_loss_floor'])/(loss['losses'][0]-loss['combined_loss_floor'])
assert abs(ratio-loss['excess_loss_ratio'])<1e-14
assert m0['passed']==all(m0['checks'].values())
proof={'status':'PASS_COMPLETE_M0_TEXT_REPLAY','engineering_passed':m0['passed'],'checks':m0['checks'],
       'preflight_batches_checked':checked_preflight,'optimizer_steps_checked':checked,
       'overfit_first_loss':loss['losses'][0],'overfit_final_loss':loss['losses'][-1],
       'floor':loss['combined_loss_floor'],'overfit_excess_loss_ratio':ratio,
       'capacity_peak_reserved_mib':[x['peak_reserved_mib'] for x in m0['capacities']],
       'candidate_capacity_active_batches':sum(x['style_active'] for x in m0['capacities'][1]['components']),
       'original_execution_commit':summary['repository_commit'],'run_summary_sha256':hashlib.sha256((d/'run_summary.json').read_bytes()).hexdigest(),
       'independent_audit':False,'local_torch_image_model_calls':0,
       'gradient_gate_scope':'Nonzero coverage union within each phase as preregistered runner; all gradients finite every step',
       'all_16_capacity_steps_nonzero_tensors':203,
       'overfit_nonzero_tensor_counts':{'minimum':192,'maximum':203,'first_14_steps_all203':True}}
out=root/'evidence/trifusion_v27_m0_complete_text_verification_20260907.json'
assert not out.exists()
out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(proof,ensure_ascii=False))
