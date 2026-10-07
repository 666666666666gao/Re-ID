from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,math
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/incremental_role_objective_m0_v1_20261007_886'
journal=root/'logs/incremental_role_objective_m0_launch_20261007_886'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((journal/'EXIT.json').read_text())['exit_code']==1
child=json.loads((journal/'CHILD.json').read_text());assert not (Path('/proc')/str(child['pid'])).exists()
state=json.loads((campaign/'campaign.json').read_text())
assert state['status']=='RUNNING' and sum('acceptance' in j for j in state['jobs'])==5
source=json.loads((root/'refine-logs/incremental_role_objective_v1/CHECKED_VJP_SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(source)==415 and all(sha(root/n)==d for n,d in source.items())
for seal_name,count in (('INPUT_SEAL.json',45),('INITIAL_BOUNDARY_INPUT_SEAL.json',48)):
    seal=json.loads((root/'refine-logs/incremental_role_objective_v1'/seal_name).read_text())
    assert len(seal['artifact_sha256'])==count and all(sha(Path(n))==d for n,d in seal['artifact_sha256'].items())
files={p for folder in (campaign,journal) for p in folder.rglob('*') if p.is_file()}
rows=[]
for job in state['jobs']:
    dataset,objective=job['dataset'],job['objective']
    assert job['status']=='COMPLETE' and job['exit_code']==0
    if 'acceptance' not in job:
        assert (dataset,objective)==('RGBNT100','repair_keep')
        continue
    acceptance=json.loads((campaign/'acceptance'/f'{dataset}_{objective}.json').read_text())
    assert acceptance==job['acceptance'] and acceptance['status']=='REAL_M0_AND_ISOLATED_INCREMENT_ACTIVITY_ACCEPTED'
    assert all(sha(Path(n))==d for n,d in acceptance['artifact_sha256'].items())
    output=Path(acceptance['output_dir']);receipt=json.loads((output/'training.json').read_text())
    assert receipt['status']=='M0_PASS' and receipt['production_m0_diagnostics']['effective_optimizer_updates']==8
    assert receipt['m0']['nonzero_gradient_parameters']==receipt['m0']['trainable_parameters']==receipt['initializer']['trainable_parameter_tensors']
    assert receipt['m0']['reload_max_abs_difference']<=1e-5
    assert all(v==8 for v in receipt['production_m0_diagnostics']['author_bn_batches_tracked'].values())
    steps=[json.loads(line) for line in (output/'training_steps.jsonl').read_text().splitlines()]
    assert len(steps)==8 and all(s['incremental_isolated_vjp_autocast_enabled'] is False for s in steps)
    assert all(s['incremental_isolated_shared_global_gradient_absent'] for s in steps)
    assert all(not u for s in steps for u in s['incremental_isolated_query_key_unused'].values())
    norms=acceptance['query_key_gradient_norm_sums']
    assert len(norms)==6 and all(math.isfinite(v) and v>0 for v in norms.values())
    assert acceptance['correction_gradient_norm_sum']>0 and not Path(acceptance['probe']['path']).exists()
    old=root/f'trained-model/global_task_role_v1_20261004_824_m0_semantic_{dataset}/training_batch_order.jsonl'
    assert (output/'training_batch_order.jsonl').read_bytes()==old.read_bytes()
    files.update(p for p in output.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'))
    rows.append(dict(dataset=dataset,objective=objective,optimizer_updates=8,
        gradient_tensors=receipt['m0']['trainable_parameters'],reload_max_abs_difference=receipt['m0']['reload_max_abs_difference'],
        correction_gradient_norm_sum=acceptance['correction_gradient_norm_sum'],query_key_gradient_norm_sums=norms,
        source_batch_order_equal=True,qualification='ACCEPTED_NEW_OBSERVATION_CONTEXT',probe_retired=True))
retirements=[json.loads(line) for line in (campaign/'probe_retirement.jsonl').read_text().splitlines()]
assert len(retirements)==5
print(json.dumps(dict(status='CHECKED_M0_MATRIX_PHYSICAL_SHA_VERIFIED_5_OF_6',at=datetime.now().astimezone().isoformat(),
    source_count=415,current45_and48_inputs_unchanged=True,rows=rows,new_qualified=5,missing_formal=1,original_qualified=0,
    total_new_optimizer_updates=48,formal_runs=0,retired_probe_count=5,
    retired_probe_bytes=sum(r['bytes'] for r in retirements),free_bytes=shutil.disk_usage(root).free,
    files={str(p.relative_to(root)):dict(text=p.read_text(),sha256=sha(p)) for p in sorted(files)},
    boundary='New five acceptances; failed100 repair_keep remains missing; with same unscaled loss/targets/nonzero rule, M0 VJP outsideautocast. Old six originalFAIL unchanged. No retrieval or full50 result; storage separately required.')))
