"""Receive actual six M0 receipts and initial/full-start records; no model work."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import paramiko

private=Path('C:/Users/gb/.codex_tmp/training_feature_scale_split755')
launch=json.loads((private/'deploy/LAUNCH.json').read_bytes())
out=private/'m0_intake757'
assert not out.exists();out.mkdir()
code='''
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path(LAUNCH['root']);campaign=Path(LAUNCH['campaign'])
state=json.loads((campaign/'campaign.json').read_bytes())
assert state['status']=='RUNNING' and state['phase']=='full'
m0=[row for row in state['jobs'] if row['phase']=='m0']
assert len(m0)==6 and all(row['status']=='COMPLETE' and row['exit_code']==0 for row in m0)
proc=Path('/proc')/str(LAUNCH['controller_pid'])
assert proc.is_dir()
cmdline=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
assert str(campaign) in cmdline and 'queue_metric_feature_scale.py' in cmdline
manifest=json.loads((campaign/'manifest.json').read_bytes())
assert manifest['schema']=='trifusion-metric-feature-scale-v1'
assert manifest['source_sha256']==LAUNCH['source_sha256'] and len(manifest['source_sha256'])==265
actual_sources={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in manifest['source_sha256']}
assert actual_sources==manifest['source_sha256']
fixed=[campaign/'manifest.json',*(campaign/'initialization').glob('*.json')]
for dataset in ('RGBNT201','RGBNT100','MSVR310'):
 for suffix in ('.json','.log'):
  fixed.append(campaign/f'initial_forward_pair_{dataset}{suffix}')
 for variant in ('normalized','metric_raw'):
  fixed.extend((campaign/f'prepare_{variant}_{dataset}.log',campaign/f'm0_{variant}_{dataset}.log'))
  child=campaign/f'{campaign.name}_m0_{variant}_{dataset}'
  fixed.extend(path for path in child.rglob('*') if path.is_file() and path.suffix in ('.json','.log'))
assert all(path.is_file() for path in fixed)
files={path.relative_to(root).as_posix():path for path in fixed}
binaries=[]
for variant in ('normalized','metric_raw'):
 for dataset in ('RGBNT201','RGBNT100','MSVR310'):
  output=root/'trained-model'/f'{campaign.name}_m0_{variant}_{dataset}'
  training=json.loads((output/'training.json').read_bytes())
  assert training['status']=='M0_PASS' and training['history'][0]['steps']==8
  assert training['m0']['nonzero_gradient_parameters']==training['m0']['trainable_parameters']
  assert training['m0']['reload_max_abs_difference']<1e-5
  for name in ('training.json','training_steps.jsonl','training_batch_order.jsonl'):
   path=output/name;files[path.relative_to(root).as_posix()]=path
  probe=output/'m0_reload_probe.pth'
  with probe.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  assert digest==training['m0']['reload_probe_sha256']
  binaries.append({'path':str(probe),'bytes':probe.stat().st_size,'sha256':digest})
text={name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in sorted(files.items())}
print(json.dumps({'status':'ALL_SIX_M0_PASS_FULL_PHASE_INTAKE','observed_at':datetime.now().astimezone().isoformat(),
 'controller_pid':LAUNCH['controller_pid'],'controller_live':True,'controller_cmdline':cmdline,
 'campaign':state,'manifest':manifest,'text':text,'binary':binaries,'source_sha256':actual_sources,
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),
 'compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True),
 'free_bytes':shutil.disk_usage(root).free,
 'boundary':'Original six M0 receipts/probe hashes and full-start state received. No model, optimizer, score, report, restart or retry executed. Binary hashes are remote attestations, not local deserialization.'}))
'''
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write('LAUNCH='+repr(launch)+'\n'+code);stdin.channel.shutdown_write();stdout.channel.settimeout(60)
data,error=stdout.read(),stderr.read();exit_code=stdout.channel.recv_exit_status()
(out/'stdout.txt').write_bytes(data);(out/'stderr.txt').write_bytes(error)
(out/'exit.json').write_text(json.dumps({'exit_code':exit_code})+'\n',encoding='utf-8')
assert exit_code==0,error.decode()
record=json.loads(data)
sftp=client.open_sftp()
for name,expected in record['text'].items():
    local=out/'raw'/name;local.parent.mkdir(parents=True,exist_ok=True)
    sftp.get(launch['root']+'/'+name,str(local))
    actual=local.read_bytes()
    assert len(actual)==expected['bytes'] and hashlib.sha256(actual).hexdigest()==expected['sha256'],name
sftp.close();client.close()
record['received_at']=datetime.now().astimezone().isoformat()
(out/'INTAKE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':record['status'],'observed_at':record['observed_at'],'text_files':len(record['text']),
                 'm0_binary_count':len(record['binary']),'free_bytes':record['free_bytes'],'gpu':record['gpu'],
                 'jobs':[{key:value for key,value in job.items() if key in ('phase','dataset','variant','status','gpu','pid','exit_code')} for job in record['campaign']['jobs']]},indent=2))
