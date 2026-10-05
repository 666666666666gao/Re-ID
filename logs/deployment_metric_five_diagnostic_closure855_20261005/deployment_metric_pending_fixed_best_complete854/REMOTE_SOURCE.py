from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
output=root/'results/deployment_metric_fixed_best_pending100_20261005_854'
launch=root/'logs/deployment_metric_pending_fixed_best_launch_20261005_854'
state=json.loads((output/'campaign.json').read_text());terminal=json.loads((launch/'EXIT.json').read_text())
assert state['status']=='COMPLETE' and terminal['exit_code']==0
assert state['optimizer_updates']==0 and state['planned_models']==2
assert [(j['dataset'],j['variant'],j['status']) for j in state['jobs']]==[
 ('RGBNT100','semantic','COMPLETE'),('RGBNT100','native','COMPLETE')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
assert all(sha(Path(n))==d for n,d in state['original_failure_immutable_sha256'].items())
seal_path=root/'refine-logs/deployment_metric_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(seal_path)==state['seal_sha256']
seal=json.loads(seal_path.read_text())
assert len(seal['source_sha256'])==341 and len(seal['artifact_sha256'])==271
assert all(sha(root/n)==d for n,d in seal['source_sha256'].items())
assert all(sha(Path(n))==d for n,d in seal['artifact_sha256'].items())
files={};reports={}
def add(p):files[p.relative_to(root).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
for j in state['jobs']:
 folder=output/(j['dataset']+'_'+j['variant']);p=folder/'DIAGNOSIS.json';r=json.loads(p.read_text())
 row=next(x for x in seal['rows'] if (x['dataset'],x['variant'])==(j['dataset'],j['variant']))
 assert r['status']=='COMPLETE' and r['model_state_before_sha256']==r['model_state_after_sha256']
 assert r['input_checkpoint_sha256']==row['checkpoint_sha256'] and r['original_receipt_sha256']==row['receipt_sha256'] and r['selected_epoch']==row['best_epoch']
 for n,info in r['artifacts'].items():assert sha(folder/n)==info['sha256'] and (folder/n).stat().st_size==info['bytes']
 assert all(abs(r['scores']['fused']['metrics'][k]-v)<1e-5 for k,v in row['metrics'].items())
 assert all(len(v['average_precision'])==len(r['comparisons']['same_model_global_to_fused']['query_changes']) for v in r['scores'].values())
 add(p);add(output/(folder.name+'.log'))
 reports[folder.name]={'readout_gain':r['readout_gain'],'metrics':{k:v['metrics'] for k,v in r['scores'].items()},
  'comparisons':{k:{n:v[n] for n in ('delta_metrics','rank1_repairs','rank1_new_errors','identity_macro_delta_ap_points')} for k,v in r['comparisons'].items()},
  'diagnostic':r['diagnostic'],'fused_distance_max_absolute_difference_from_original':r['fused_distance_max_absolute_difference_from_original'],
  'elapsed_seconds':r['elapsed_seconds'],'input_checkpoint_sha256':r['input_checkpoint_sha256']}
for p in (output/'campaign.json',launch/'LAUNCH.json',launch/'EXIT.json',launch/'supervisor.py',launch/'supervisor.log',launch/'console.log'):add(p)
assert not Path('/proc/'+str(json.loads((launch/'LAUNCH.json').read_text())['pid'])+'/stat').exists()
print(json.dumps(dict(status='TWO_FIRST100_FIXED_DIAGNOSES_COMPLETE_ORIGINAL_FAILURE_UNCHANGED',at=datetime.now().astimezone().isoformat(),
 campaign=state,exit=terminal,reports=reports,files=files,source_files_verified=341,artifact_files_verified=271,optimizer_updates=0,
 original_failed_parent_and_two201_diagnoses_sha_unchanged=True,disk_free_bytes=shutil.disk_usage(root).free)))
