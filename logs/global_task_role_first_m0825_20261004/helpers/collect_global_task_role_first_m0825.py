"""Collect the completed first M0 and a bounded live-training snapshot."""
from datetime import datetime
import json
from pathlib import Path
import paramiko

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/global_task_role_first_m0825')
assert not packet.exists()
code='''
from datetime import datetime
from pathlib import Path
import hashlib,json,subprocess,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/global_task_role_v1_20261004_824'
launch=root/'logs/global_task_role_launch_20261004_824'
state=json.loads((campaign/'campaign.json').read_text())
assert state['status']=='RUNNING' and state['report_invocations']==0
job=next(j for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==('RGBNT201','semantic','m0'))
assert job['status']=='COMPLETE' and job['exit_code']==0
m0root=root/'trained-model/global_task_role_v1_20261004_824_m0_semantic_RGBNT201'
m0=json.loads((m0root/'training.json').read_text())
assert m0['status']=='M0_PASS' and m0['schema']=='trifusion-global-task-role-v1'
assert m0['history'][0]['steps']==8
assert m0['m0']['nonzero_gradient_parameters']==m0['m0']['trainable_parameters']
assert m0['m0']['reload_max_abs_difference']<=1e-5
assert hashlib.sha256((m0root/'m0_reload_probe.pth').read_bytes()).hexdigest()==m0['m0']['reload_probe_sha256']
assert m0['production_m0_diagnostics']['effective_optimizer_updates']==8
assert all(n==8 for n in m0['production_m0_diagnostics']['author_bn_batches_tracked'].values())
initializer=campaign/'initialization/RGBNT201_semantic.json'
init=json.loads(initializer.read_text())
controls=json.loads((root/'refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
old=next(r['initializer'] for r in controls['rows'] if (r['dataset'],r['variant'])==('RGBNT201','semantic'))
excluded=('architecture','entry_sha256','scope','objective_gradient_policy')
assert {k:v for k,v in init['binding'].items() if k not in excluded}=={k:v for k,v in old.items() if k not in excluded}
scope=json.loads((root/'refine-logs/global_task_role_v1/SOURCE_SCOPE.json').read_text())
assert len(scope['source_sha256'])==330
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in scope['source_sha256'].items())
active=state['active_command']
assert active['mode']=='train' and active['status']=='RUNNING'
process=Path('/proc/'+str(active['pid']))
stat=(process/'stat').read_text()
assert 'run_global_task_role.py' in (process/'cmdline').read_bytes().decode().replace('\\x00',' ')
epochs=[]
for line in (campaign/'RGBNT201_semantic_train.log').read_text().splitlines():
 if line.startswith('{'):
  item=json.loads(line)
  if item.get('event')=='epoch':epochs.append(item)
textpaths=[campaign/'manifest.json',campaign/'initialization/RGBNT201_semantic.json',m0root/'training.json',
 campaign/'prepare_RGBNT201_semantic.log',campaign/'RGBNT201_semantic_m0.log',launch/'LAUNCH.json',launch/'supervisor.py']
files={str(p.relative_to(root)):p.read_text() for p in textpaths}
summary=dict(status='FIRST_REAL_M0_PASS_FRESH50_LIVE',at=datetime.now().astimezone().isoformat(),
 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 source_files_verified=330,dataset='RGBNT201',variant='semantic',m0=m0['m0'],
 effective_optimizer_updates=m0['production_m0_diagnostics']['effective_optimizer_updates'],
 author_bn_batches_tracked=m0['production_m0_diagnostics']['author_bn_batches_tracked'],
 initialization_matches_detached_control=True,matched_fields_exclusions=list(excluded),
 initialization_sha256=hashlib.sha256(initializer.read_bytes()).hexdigest(),active_process_pid=active['pid'],
 active_process_stat=stat,formal_completed_epochs=len(epochs),formal_closed_epoch_logs=epochs,
 formal_acceptance_complete=False,completed_m0=1,formal_accepted=0,
 disk_free_bytes=shutil.disk_usage(root).free,temperature_power_queried=False)
print(json.dumps(dict(summary=summary,files=files)))
'''
compile(code,'remote_first_m0_collector.py','exec')
packet.mkdir()
(packet/'remote_first_m0_collector.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
out.channel.settimeout(180)
data,error=out.read(),err.read()
status=out.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data)
(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status,at=datetime.now().astimezone().isoformat()))+'\n',encoding='utf-8')
client.close()
assert status==0,error.decode()
received=json.loads(data)
for name,value in received['files'].items():
    p=packet/'received'/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(value,encoding='utf-8')
(packet/'SUMMARY.json').write_text(json.dumps(received['summary'],indent=2)+'\n',encoding='utf-8')
print(json.dumps(received['summary']))
