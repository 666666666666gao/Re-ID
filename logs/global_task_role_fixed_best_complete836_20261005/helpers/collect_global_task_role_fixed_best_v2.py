from pathlib import Path
import hashlib
import json
import paramiko

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/global_task_role_fixed_best_complete')
assert not packet.exists()
packet.mkdir()
code='''
from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
output=root/'results/global_task_role_fixed_best_20261004_v1'
launch=root/'logs/global_task_role_fixed_best_launch_20261004_v1'
state=json.loads((output/'campaign.json').read_text())
exit=json.loads((launch/'EXIT.json').read_text())
assert state['status']=='COMPLETE' and exit['exit_code']==0
assert len(state['jobs'])==6 and all(r['status']=='COMPLETE' and r['exit_code']==0 for r in state['jobs'])
sealpath=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
seal=json.loads(sealpath.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(sealpath)==state['seal_sha256']
assert all(sha(root/name)==digest for name,digest in seal['source_sha256'].items())
assert all(sha(Path(name))==digest for name,digest in seal['artifact_sha256'].items())
files={}
def add(p):
 files[str(p.relative_to(root))]={'sha256':sha(p),'bytes':p.stat().st_size}
reports={}
for job in state['jobs']:
 folder=output/(job['dataset']+'_'+job['variant'])
 p=folder/'DIAGNOSIS.json';r=json.loads(p.read_text())
 assert r['status']=='COMPLETE' and r['model_state_before_sha256']==r['model_state_after_sha256']
 assert (r['dataset'],r['variant'])==(job['dataset'],job['variant'])
 for name,record in r['artifacts'].items():
  assert (folder/name).stat().st_size==record['bytes'] and sha(folder/name)==record['sha256']
 assert all(len(r['scores'][key]['average_precision'])==len(r['comparisons']['same_model_global_to_fused']['query_changes']) for key in r['scores'])
 assert all(abs(r['comparisons']['independent_global_only_to_same_model_global']['delta_metrics'][key]
  +r['comparisons']['same_model_global_to_fused']['delta_metrics'][key]
  -r['comparisons']['independent_global_only_to_fused']['delta_metrics'][key])<1e-10 for key in r['scores']['fused']['metrics'])
 add(p);add(output/(job['dataset']+'_'+job['variant']+'.log'))
 reports[p.parent.name]={'readout_gain':r['readout_gain'],'metrics':{k:v['metrics'] for k,v in r['scores'].items()},
  'comparisons':{k:{name:v[name] for name in ('delta_metrics','rank1_repairs','rank1_new_errors','identity_macro_delta_ap_points')} for k,v in r['comparisons'].items()},
  'diagnostic':r['diagnostic'],'fused_distance_max_absolute_difference_from_original':r['fused_distance_max_absolute_difference_from_original'],
  'elapsed_seconds':r['elapsed_seconds'],'input_checkpoint_sha256':r['input_checkpoint_sha256']}
for p in (output/'campaign.json',launch/'LAUNCH.json',launch/'EXIT.json',launch/'supervisor.py',launch/'supervisor.log',launch/'console.log'):
 add(p)
proc=Path('/proc/'+str(json.loads((launch/'LAUNCH.json').read_text())['pid'])+'/stat')
assert not proc.exists()
print(json.dumps({'status':'SIX_FIXED_BEST_READ_ONLY_DIAGNOSES_VERIFIED','at':datetime.now().astimezone().isoformat(),
 'campaign':state,'launch_exit':exit,'source_files_verified':len(seal['source_sha256']),
 'original_artifacts_verified':len(seal['artifact_sha256']),'optimizer_updates':0,
 'all_model_parameter_and_buffer_sha_unchanged':True,'all_original_formal_inputs_sha_unchanged':True,
 'all_fused_metric_replays_passed_fixed_tolerance':True,'reports':reports,'files':files,
 'disk_free_bytes':shutil.disk_usage(root).free}))
'''
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(300)
data,error=stdout.read(),stderr.read();exit_code=stdout.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps({'exit_code':exit_code})+'\n',encoding='utf-8')
assert exit_code==0,error.decode()
record=json.loads(data)
texts=packet/'texts';texts.mkdir()
sftp=client.open_sftp()
for name,info in record['files'].items():
 target=texts/(Path(name).parent.name+'_'+Path(name).name)
 assert not target.exists()
 sftp.get('/data/gaob/Re-ID/Trifusion/'+name,str(target))
 assert target.stat().st_size==info['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==info['sha256']
sftp.close();client.close()
print(json.dumps({'status':record['status'],'at':record['at'],'reports':record['reports'],
 'source_files_verified':record['source_files_verified'],'original_artifacts_verified':record['original_artifacts_verified'],
 'disk_free_bytes':record['disk_free_bytes']}))
