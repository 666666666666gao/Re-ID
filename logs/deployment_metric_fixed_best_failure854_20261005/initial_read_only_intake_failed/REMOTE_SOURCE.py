from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
output=root/'results/deployment_metric_fixed_best_five_20261005_850'
launch=root/'logs/deployment_metric_fixed_best_five_launch_20261005_850'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
state=json.loads((output/'campaign.json').read_text());terminal=json.loads((launch/'EXIT.json').read_text())
assert state['status']=='FAILED' and terminal['exit_code']==1
assert [(j['dataset'],j['variant'],j['status']) for j in state['jobs']]==[
 ('RGBNT201','semantic','COMPLETE'),('RGBNT201','native','COMPLETE'),('MSVR310','semantic','FAILED')]
assert state['optimizer_updates']==0
seal_path=root/'refine-logs/deployment_metric_fixed_best_diagnosis_v1/INPUT_SEAL.json'
seal=json.loads(seal_path.read_text());assert sha(seal_path)==state['seal_sha256']
assert len(seal['source_sha256'])==341 and len(seal['artifact_sha256'])==271
assert all(sha(root/n)==d for n,d in seal['source_sha256'].items())
assert all(sha(Path(n))==d for n,d in seal['artifact_sha256'].items())
assert not Path('/proc/'+str(json.loads((launch/'LAUNCH.json').read_text())['pid'])+'/stat').exists()
files={};reports={}
def add(p):files[p.relative_to(root).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
for j in state['jobs']:
 folder=output/(j['dataset']+'_'+j['variant']);add(output/(folder.name+'.log'))
 if j['status']=='COMPLETE':
  p=folder/'DIAGNOSIS.json';r=json.loads(p.read_text());assert r['status']=='COMPLETE'
  assert r['model_state_before_sha256']==r['model_state_after_sha256']
  row=next(x for x in seal['rows'] if (x['dataset'],x['variant'])==(j['dataset'],j['variant']))
  assert r['input_checkpoint_sha256']==row['checkpoint_sha256']
  assert r['original_receipt_sha256']==row['receipt_sha256'] and r['selected_epoch']==row['best_epoch']
  for n,info in r['artifacts'].items():assert sha(folder/n)==info['sha256'] and (folder/n).stat().st_size==info['bytes']
  add(p)
  reports[folder.name]={'readout_gain':r['readout_gain'],'metrics':{k:v['metrics'] for k,v in r['scores'].items()},
   'diagnostic':r['diagnostic'],'fused_distance_max_absolute_difference_from_original':r['fused_distance_max_absolute_difference_from_original'],
   'comparisons':{k:{n:v[n] for n in ('delta_metrics','rank1_repairs','rank1_new_errors','identity_macro_delta_ap_points')} for k,v in r['comparisons'].items()},
   'elapsed_seconds':r['elapsed_seconds'],'input_checkpoint_sha256':r['input_checkpoint_sha256']}
 else:
  assert not (folder/'DIAGNOSIS.json').exists()
  assert not (folder/'query_features.pt').exists() and not (folder/'gallery_features.pt').exists()
for p in (output/'campaign.json',launch/'LAUNCH.json',launch/'EXIT.json',launch/'supervisor.py',launch/'supervisor.log',launch/'console.log'):add(p)
print(json.dumps(dict(status='ORIGINAL_FIXED_DIAGNOSIS_FAILURE_AND_TWO_COMPLETED_INPUTS_SEALED',at=datetime.now().astimezone().isoformat(),
 campaign=state,exit=terminal,files=files,reports=reports,source_files_verified=341,artifact_files_verified=271,
 optimizer_updates=0,missing_dependency_exists=(root/'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json').exists(),
 failed_log=(output/'MSVR310_semantic.log').read_text(),disk_free_bytes=shutil.disk_usage(root).free)))
