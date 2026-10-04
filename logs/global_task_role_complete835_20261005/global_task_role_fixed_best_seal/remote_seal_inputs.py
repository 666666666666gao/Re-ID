
from datetime import datetime
from pathlib import Path
import hashlib,json,shutil,subprocess

root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/global_task_role_v1_20261004_824'
report=root/'results/global_task_role_v1_complete_20261004_824'
launch=root/'logs/global_task_role_launch_20261004_824'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
state=json.loads((campaign/'campaign.json').read_text())
exit_record=json.loads((launch/'EXIT.json').read_text())
assert exit_record['exit_code']==0
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==0
assert len(state['jobs'])==12 and all(row['status']=='COMPLETE' and row['exit_code']==0 for row in state['jobs'])
matrix=json.loads((campaign/'accepted_matrix.json').read_text())
summary=json.loads((report/'SUMMARY.json').read_text())
assert matrix['schema']==summary['schema']=='trifusion-global-task-role-v1'
assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==summary['accepted']==6
assert summary['status']=='COMPLETE' and summary['formal_epochs']==300 and summary['formal_steps']==12968
assert len(summary['pairs'])==18
manifest=json.loads((campaign/'manifest.json').read_text())
scope=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/SOURCE_SCOPE.json').read_text())
source=scope['source_sha256']
assert len(source)==332 and len(manifest['source_sha256'])==330
assert set(source)-set(manifest['source_sha256'])=={'tools/diagnose_global_task_role_best.py','refine-logs/global_task_role_fixed_best_diagnosis_v1/EXPERIMENT_PLAN.md'}
assert all(source[name]==digest for name,digest in manifest['source_sha256'].items())
assert all(sha(root/name)==digest for name,digest in source.items())
assert all(sha(Path(name))==digest for name,digest in manifest['initialization_sha256'].items())
controls_path=root/'refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json'
controls=json.loads(controls_path.read_text())
assert len(controls['rows'])==9 and len(controls['artifact_sha256'])==124
assert all(sha(Path(name))==digest for name,digest in controls['artifact_sha256'].items())
rows=list(matrix['rows'])+[row for row in controls['rows'] if row['variant']=='global_only']
assert {(row['dataset'],row['variant']) for row in rows}=={(dataset,variant) for dataset in ('RGBNT201','MSVR310','RGBNT100') for variant in ('global_only','semantic','native')}
artifacts=dict(controls['artifact_sha256'])
artifacts[str(controls_path)]=sha(controls_path)
for base,names in ((campaign,('accepted_matrix.json','manifest.json','campaign.json','report.log')),
                   (report,('SUMMARY.json','REPORT.md')),
                   (launch,('LAUNCH.json','EXIT.json'))):
    for name in names:
        artifacts[str(base/name)]=sha(base/name)
artifacts.update(manifest['initialization_sha256'])
for row in matrix['rows']:
    assert row['status']=='VERIFIED_COMPLETE'
    accepted=next(item for item in summary['rows'] if (item['dataset'],item['variant'])==(row['dataset'],row['variant']))
    assert all(accepted[key]==value for key,value in row.items())
    out=Path(row['run_dir'])
    receipt=json.loads((out/'official_metrics.json').read_text())
    training=json.loads((out/'training.json').read_text())
    assert receipt['status']=='COMPLETE' and receipt['metrics']==row['metrics']
    assert len(training['history'])==50 and training['best_epoch']==row['best_epoch']
    assert [path.name for path in out.glob('*.pth')]==['best_map.pth']
    for name,key in (('best_map.pth','checkpoint_sha256'),('official_distances.pt','distance_sha256'),('official_metrics.json','receipt_sha256')):
        assert sha(out/name)==row[key]
        artifacts[str(out/name)]=row[key]
    assert sha(out/'best_epoch_distances.pt')==receipt['training_best_distance_sha256']
    for name in ('training.json','best_epoch_distances.pt','training_steps.jsonl','training_batch_order.jsonl'):
        artifacts[str(out/name)]=sha(out/name)
    original=next(item for item in controls['rows'] if (item['dataset'],item['variant'])==(row['dataset'],row['variant']))
    assert (out/'training_batch_order.jsonl').read_bytes()==(Path(original['run_dir'])/'training_batch_order.jsonl').read_bytes()
    m0=Path(row['m0_dir'])/'training.json'
    artifacts[str(m0)]=sha(m0)
assert all(sha(Path(name))==digest for name,digest in artifacts.items())
print(json.dumps({'status':'ALL_SIX_FULL50_FIRST_STRICT_AND_ONCE_CPU_REPORT_INPUTS_VERIFIED',
    'at':datetime.now().astimezone().isoformat(),'source_sha256':source,'artifact_sha256':artifacts,'rows':rows,
    'source_count':len(source),'original_control_artifacts':124,'original_campaign':'logs/global_task_role_v1_20261004_824',
    'original_once_cpu_report':'results/global_task_role_v1_complete_20261004_824',
    'disk_free_bytes':shutil.disk_usage(root).free,'remote_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    'boundary':'Input validation only; no model construction/inference, training, old M0 verifier or report replay. New M0 receipt sealed; probe binary is not a diagnosis dependency. Fixed selected best and original controls unchanged; Goal active/unmet.'}))
