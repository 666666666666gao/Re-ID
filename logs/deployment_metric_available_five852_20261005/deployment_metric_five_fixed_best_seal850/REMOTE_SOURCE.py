
from datetime import datetime
from pathlib import Path
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion')
origin=root/'logs/deployment_metric_role_v1_20261005_837'
pending=root/'logs/deployment_metric_role_pending100_20261005_842'
continuation=root/'logs/deployment_metric_storage_continuation_20261005_848'
report=root/'results/deployment_metric_storage_five_20261005_848'
report_launch=root/'logs/deployment_metric_storage_five_report_launch_20261005_850'
expected=[('RGBNT201','semantic'),('RGBNT201','native'),('MSVR310','semantic'),('RGBNT100','semantic'),('RGBNT100','native')]
new_sources={'refine-logs/deployment_metric_fixed_best_diagnosis_v1/DIAGNOSE_FIVE.py': 'f191b367fe4aa63dd8c45df57bd0f4ebf053a9cf75093211d6796bd110998c7d', 'refine-logs/deployment_metric_fixed_best_diagnosis_v1/EXPERIMENT_PLAN.md': 'b13f56affe0815f1ddc5f8b4468f5bf009b0ef263a7b7dcb575106f3472c8dd4'}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
assert sha(origin/'campaign.json')=='88e2575a7d3693afc0e9e02d0ed3352e61e0eb02447263b8a5c4ff1612db62f9'
assert sha(pending/'campaign.json')=='4d5b9e00e2b3739a59c0ae411c89b28b4955701b5bb16780eaab84e54130223d'
assert json.loads((root/'logs/deployment_metric_storage_launch_20261005_848/EXIT.json').read_text())['exit_code']==0
state=json.loads((continuation/'campaign.json').read_text())
assert state['status']=='COMPLETE' and 'active_command' not in state
assert len(state['jobs'])==3 and all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
assert state['report_invocations']==0
assert json.loads((report_launch/'EXIT.json').read_text())['exit_code']==0
summary=json.loads((report/'SUMMARY.json').read_text())
assert summary['status']=='PARTIAL_FIVE_FORMAL_ENDPOINTS_AND_ONE_ORIGINAL_M0_FAILURE'
assert summary['report_work_status']=='COMPLETE' and summary['accepted_formal_endpoints']==5
assert summary['formal_epochs']==250 and summary['formal_steps']==12262
assert len(summary['pairs'])==12 and len(summary['unavailable_pairs'])==3
assert [(r['dataset'],r['variant']) for r in summary['rows']]==expected
producer=root/'refine-logs/deployment_metric_role_v1/REPORT_STORAGE_FIVE_848.py'
assert sha(producer)==summary['producer_sha256']=='545c115b3402337285c88e8870e961f7fba36eeae7055fba4c92d00e169755b2'
scope_path=root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json'
assert sha(scope_path)=='76987d2ca7d6dc7a9aeac34000d37ebe5e7d1d8ead6d52b1832f0b1557523a3d'
sources=json.loads(scope_path.read_text())['source_sha256']
assert len(sources)==339 and all(sha(root/name)==digest for name,digest in sources.items())
assert not (set(new_sources)&set(sources))
assert all(sha(root/name)==digest for name,digest in new_sources.items())
sources.update(new_sources)
assert len(sources)==341
controls_path=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(controls_path)=='9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
controls=json.loads(controls_path.read_text())
assert len(controls['rows'])==9 and len(controls['artifact_sha256'])==187
assert all(sha(Path(name))==digest for name,digest in controls['artifact_sha256'].items())
artifacts=dict(controls['artifact_sha256'])
artifacts[str(controls_path)]=sha(controls_path)
artifacts[str(scope_path)]=sha(scope_path)
for folder,names in ((origin,('campaign.json','manifest.json','probe_retirement.jsonl')),
 (pending,('campaign.json','manifest.json','probe_retirement.jsonl')),
 (continuation,('campaign.json','manifest.json','accepted_matrix.json','probe_retirement.jsonl')),
 (report,('SUMMARY.json','REPORT.md')),(report_launch,('LAUNCH.json','EXIT.json','report.log'))):
 for name in names:
  artifacts[str(folder/name)]=sha(folder/name)
old_exit=root/'logs/deployment_metric_pending100_launch_20261005_843/EXIT.json'
assert sha(old_exit)=='57418a8ba131304cf38feb34195a47da211e2a15ffbc023bccbb5006591a2aac'
artifacts[str(old_exit)]=sha(old_exit)
rows=[]
for dataset,variant in expected:
 campaign=(pending if variant=='semantic' else continuation) if dataset=='RGBNT100' else origin
 receipt_path=campaign/'acceptance'/f'{dataset}_{variant}.json'
 receipt=json.loads(receipt_path.read_text())
 assert receipt['status']=='FULL50_FIRST_STRICT_AND_M0_VERIFIED_BEFORE_RETIREMENT'
 assert all(sha(Path(name))==digest for name,digest in receipt['artifact_sha256'].items())
 assert not Path(receipt['probe']['path']).exists()
 retirement=[r for r in map(json.loads,(campaign/'probe_retirement.jsonl').read_text().splitlines()) if (r['dataset'],r['variant'])==(dataset,variant)]
 assert len(retirement)==1 and retirement[0]['acceptance_sha256']==sha(receipt_path)
 assert all(retirement[0][k]==v for k,v in receipt['probe'].items())
 row=receipt['row']
 recorded=next(r for r in summary['rows'] if (r['dataset'],r['variant'])==(dataset,variant))
 assert all(recorded[k]==v for k,v in row.items())
 run=Path(row['run_dir'])
 training=json.loads((run/'training.json').read_text())
 official=json.loads((run/'official_metrics.json').read_text())
 assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and official['status']=='COMPLETE'
 assert [r['epoch'] for r in training['history']]==list(range(1,51))
 assert training['best_epoch']==official['selected_epoch']==row['best_epoch']
 assert official['metrics']==row['metrics'] and official['checkpoint_sha256']==sha(run/'best_map.pth')==row['checkpoint_sha256']
 assert official['distance_sha256']==sha(run/'official_distances.pt')==row['distance_sha256']
 assert sha(run/'official_metrics.json')==row['receipt_sha256']
 assert {p.name for p in run.glob('*.pth')}=={'best_map.pth'}
 witness_path=campaign/'initialization'/f'{dataset}_{variant}.json'
 assert json.loads(witness_path.read_text())['binding']==row['initializer']==training['initializer']
 control=next(r for r in controls['rows'] if (r['dataset'],r['variant'])==(dataset,variant))
 assert (run/'training_batch_order.jsonl').read_bytes()==(Path(control['run_dir'])/'training_batch_order.jsonl').read_bytes()
 artifacts.update(receipt['artifact_sha256'])
 artifacts[str(receipt_path)]=sha(receipt_path)
 artifacts[str(witness_path)]=sha(witness_path)
 rows.append(dict(row,artifact_campaign=str(campaign)))
rows.extend(r for r in controls['rows'] if r['variant']=='global_only')
assert len(rows)==8
assert not any((r['dataset'],r['variant'])==('MSVR310','native') for r in rows)
assert all(sha(root/name)==digest for name,digest in sources.items())
assert all(sha(Path(name))==digest for name,digest in artifacts.items())
seal=dict(schema='trifusion-deployment-metric-role-five-fixed-best-diagnosis-v1',status='FIVE_AVAILABLE_FULL50_FIRST_STRICT_AND_ORIGINAL_CPU_REPORT_INPUTS_VERIFIED',
 registered_at=datetime.now().astimezone().isoformat(),source_sha256=sources,artifact_sha256=artifacts,rows=rows,
 original_saved_array_report=str(report),source_count=341,scientific_source_count=339,original_control_artifacts=187,
 planned_models=5,physical_gpus=[0,1],optimizer_updates=0,
 boundary='Five accepted fixed selected bests across three actual campaigns plus three independent global controls. Original MSVRnative M0 failure and disk parent EXIT1 unchanged. Input sealing only; no model construction/inference, training, retired probe verifier, checkpoint reselection, report replay or power/temperature action. Full goal remains active/unmet.')
journal=root/'logs/deployment_metric_five_fixed_best_seal_20261005_850'
assert not journal.exists()
journal.mkdir()
(journal/'INPUT_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
print(json.dumps(seal))
