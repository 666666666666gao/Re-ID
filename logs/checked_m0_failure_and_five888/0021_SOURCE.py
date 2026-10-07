from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m0=root/'logs/incremental_role_objective_m0_v1_20261007_886'
journal=root/'logs/incremental_role_objective_m0_launch_20261007_886'
state=json.loads((m0/'campaign.json').read_text())
assert state['status']=='RUNNING' and sum('acceptance' in j for j in state['jobs'])==5
assert json.loads((journal/'EXIT.json').read_text())['exit_code']==1
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and (proc/'cmdline').is_file():
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in (proc/'cmdline').read_bytes().split(bytes([0])))
scope=json.loads((root/'refine-logs/incremental_role_objective_v1/CHECKED_VJP_SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(scope)==415 and all(sha(root/n)==d for n,d in scope.items())
protected={}
for name in ('INPUT_SEAL.json','INITIAL_BOUNDARY_INPUT_SEAL.json'):
    protected.update(json.loads((root/'refine-logs/incremental_role_objective_v1'/name).read_text())['artifact_sha256'])
assert all(sha(Path(n))==d for n,d in protected.items())
controls=json.loads((root/'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json').read_text())
global_metrics={r['dataset']:r['metrics'] for r in controls['rows'] if r['variant']=='global_only'}
rows=[]
specs=[('global_task_role_v1_20261004_824','raw_native'),('semantic_capacity_control_v1_20261005_856','semantic_capacity')]
for family,label in specs:
    for dataset in ('RGBNT201','MSVR310','RGBNT100'):
        folder=root/'trained-model'/f'{family}_full_native_{dataset}'
        training=json.loads((folder/'training.json').read_text());official=json.loads((folder/'official_metrics.json').read_text())
        assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and official['status']=='COMPLETE'
        assert [r['epoch'] for r in training['history']]==list(range(1,51))
        assert training['best_epoch']==official['selected_epoch']
        assert official['independent_upstream_metrics_equal'] and not official['reranking']
        assert official['metrics']['mAP']-global_metrics[dataset]['mAP']<0.5
        best=folder/'best_map.pth'
        assert best.resolve().is_relative_to((root/'trained-model').resolve()) and best.stat().st_nlink==1
        assert str(best) not in protected and str(best.relative_to(root)) not in scope
        assert sha(best)==official['checkpoint_sha256']
        assert sha(folder/'official_distances.pt')==official['distance_sha256']
        assert sha(folder/'best_epoch_distances.pt')==official['training_best_distance_sha256']
        if label=='raw_native':
            diagnosis=root/f'results/global_task_role_fixed_best_20261004_v1/{dataset}_native/DIAGNOSIS.json'
            d=json.loads(diagnosis.read_text())
            assert d['status']=='COMPLETE' and d['input_checkpoint_sha256']==sha(best)
        rows.append(dict(path=str(best),bytes=best.stat().st_size,sha256=sha(best),label=label,dataset=dataset,
            selected_epoch=official['selected_epoch'],metrics=official['metrics'],
            retained_artifact_sha256={str(p):sha(p) for p in folder.iterdir() if p.is_file() and p!=best},
            reason='Closed non-advancing extra-source route; all declared fixed-model consumers terminal. Current matched semantic/global controls and strong/winning routes retained. Exact best/official arrays, trajectories, receipts remain. No pending model consumer in semantic-only increment experiment.'))
capacity_state=json.loads((root/'logs/semantic_capacity_first_eval_completion_20261005_857/campaign.json').read_text())
assert capacity_state['status']=='COMPLETE' and capacity_state['report_exit_code']==0
capacity_report=json.loads((root/'results/semantic_capacity_control_v1_complete_20261005_856/SUMMARY.json').read_text())
assert capacity_report['status']=='COMPLETE' and capacity_report['accepted']==3 and len(capacity_report['pairs'])==9
raw_state=json.loads((root/'logs/global_task_role_v1_20261004_824/campaign.json').read_text())
assert raw_state['status']=='COMPLETE' and raw_state['report_exit_code']==0
fixed=root/'results/incremental_fixed_m0_scale_20261007_879/DIAGNOSIS.json'
assert json.loads(fixed.read_text())['status']=='FIXED_M0_SCALE_COMPARISON_COMPLETE'
assert json.loads((root/'logs/incremental_fixed_m0_scale_launch_20261007_879/EXIT.json').read_text())['exit_code']==0
probe=root/'trained-model/incremental_role_objective_m0_v1_20261007_878_m0_md_batch_ratio_RGBNT201/m0_reload_probe.pth'
receipt=json.loads((probe.parent/'training.json').read_text())
assert probe.resolve().is_relative_to((root/'trained-model').resolve()) and probe.stat().st_nlink==1
assert str(probe) not in protected and str(probe.relative_to(root)) not in scope
assert sha(probe)==receipt['m0']['reload_probe_sha256']
rows.append(dict(path=str(probe),bytes=probe.stat().st_size,sha256=sha(probe),label='closed_original_M0_probe',
    retained_artifact_sha256={str(p):sha(p) for p in probe.parent.iterdir() if p.is_file() and p!=probe},
    reason='Original failure retained. Sole post-eight fixed-probe consumer completed; fresh-initial/context/checked stages use no probe. Historical fixed9 binary replay retires only here, not retroactive acceptance.'))
failed_probe=root/'trained-model/incremental_role_objective_m0_v1_20261007_886_m0_repair_keep_RGBNT100/m0_reload_probe.pth'
failed_receipt=json.loads((failed_probe.parent/'training.json').read_text())
assert failed_receipt['status']=='M0_PASS' and failed_receipt['production_m0_diagnostics']['effective_optimizer_updates']==8
failed_steps=[json.loads(line) for line in (failed_probe.parent/'training_steps.jsonl').read_text().splitlines()]
assert len(failed_steps)==8
for kind in ('query','key'):
    assert sum(s['incremental_isolated_query_key_gradient_norms'][f'evidence_model.roles.{kind}_projections.2.weight'] for s in failed_steps)==0
assert sha(failed_probe)==failed_receipt['m0']['reload_probe_sha256']
assert failed_probe.resolve().is_relative_to((root/'trained-model').resolve()) and failed_probe.stat().st_nlink==1
assert str(failed_probe) not in protected and str(failed_probe.relative_to(root)) not in scope
rows.append(dict(path=str(failed_probe),bytes=failed_probe.stat().st_size,sha256=sha(failed_probe),label='closed_new_failed100_M0_probe',
    retained_artifact_sha256={str(p):sha(p) for p in failed_probe.parent.iterdir() if p.is_file() and p!=failed_probe},
    reason='Failed auxiliary gate preserved, no new probe consumer or formal100repair stage. Complete eightstep production/isolated observations/reload evidence kept. No retry/gate reclassification.'))
value=dict(status='EIGHT_CLOSED_OWN_WEIGHTS_QUALIFIED_NOT_RETIRED',at=datetime.now().astimezone().isoformat(),rows=rows,
    count=len(rows),bytes=sum(r['bytes'] for r in rows),free_bytes=shutil.disk_usage(root).free,
    required_full_bytes=5192548352,projected_free_bytes=shutil.disk_usage(root).free+sum(r['bytes'] for r in rows),
    current45_and48_and415_unchanged=True,
    boundary='Read-only eight explicit OWN candidates. Raw-native3, semantic-capacity3, original failed-M0 probe1 plus new failed100 probe1 after all consumers closed. No authors/public/current semantic-global controls/strong winners/other-project file. Historical187/241/fixed9 PTH replay will need regeneration; primary evidence and all best/official distances remain.')
print(json.dumps(value))
