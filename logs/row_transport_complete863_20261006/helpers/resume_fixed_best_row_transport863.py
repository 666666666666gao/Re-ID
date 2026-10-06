"""Once-only resume of two missing fixed-best endpoints, retaining prior failure."""
from datetime import datetime
import hashlib,json
from pathlib import Path
import paramiko

HOME=Path('C:/Users/gb')
BASE=HOME/'.codex_tmp/independent_evidence_draft'
PACKAGE=BASE/'fixed_best_row_transport_diagnosis_v1'
LOCAL=BASE/'fixed_best_row_transport_resume863'
ROOT='/data/gaob/Re-ID/Trifusion'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
LAUNCH=ROOT+'/logs/fixed_best_row_transport_resume_20261006_863'
DIAGNOSIS=ROOT+'/logs/fixed_best_row_transport_diagnosis_20261006_862'
REPORT=ROOT+'/results/fixed_best_row_transport_diagnosis_20261006_862'
OLD=ROOT+'/logs/fixed_best_row_transport_launch_20261006_862'
NAME='tools/diagnose_row_mass_fixed_best_single.py'


def main():
    verified=json.loads((BASE/'fixed_best_cfg_carryover862_v2.json').read_text())
    assert verified['exit_code']==0
    assert json.loads(verified['stdout'])['fresh_cfg_exactly_matches_initializer']
    assert not LOCAL.exists()
    LOCAL.mkdir()
    digest=hashlib.sha256((PACKAGE/NAME).read_bytes()).hexdigest()
    settings=dict(root=ROOT,python=PYTHON,launch=LAUNCH,diagnosis=DIAGNOSIS,
        report=REPORT,old=OLD,source=NAME,source_sha256=digest)
    source='settings='+repr(settings)+'\n'+'''
from datetime import datetime
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path(settings['root']);old=Path(settings['old']);diag=Path(settings['diagnosis'])
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
old_launch=json.loads((old/'LAUNCH.json').read_text())
old_exit=json.loads((old/'EXIT.json').read_text())
assert old_exit['exit_code']==1 and old_exit['stage']=='diagnose'
assert not (Path('/proc')/str(old_launch['pid'])/'stat').exists()
assert not (Path('/proc')/str(old_exit['steps'][0]['pid'])/'stat').exists()
assert not Path(settings['launch']).exists() and not Path(settings['report']).exists()
assert not (root/settings['source']).exists()
seal=json.loads((diag/'INPUT_SEAL.json').read_text())
assert all(sha(root/n)==h for n,h in seal['original_source_sha256'].items())
assert all(sha(root/n)==h for n,h in seal['diagnostic_source_sha256'].items())
matrix=json.loads((Path(seal['original_campaign'])/'accepted_matrix.json').read_text())
assert matrix['accepted']==6
assert not (diag/'SUMMARY.json').exists()
for row in matrix['rows']:
 folder=diag/(row['dataset']+'_'+row['variant'])
 if row['dataset']=='RGBNT100':
  assert not folder.exists()
 else:
  endpoint=json.loads((folder/'ENDPOINT.json').read_text())
  assert len(endpoint['modes'])==3
  for mode in endpoint['modes']:
   directory=folder/mode['mode']
   assert sha(directory/'distances.pt')==mode['distance_sha256']
   assert sha(directory/'diagnostic_arrays.pt')==mode['arrays_sha256']
  assert all(abs(endpoint['modes'][0]['metrics'][k]-row['metrics'][k])<1e-5 for k in row['metrics'])
assert shutil.disk_usage(root).free>=3*1024**3
gpu=subprocess.check_output(['nvidia-smi','-i','0,1','--query-gpu=index,memory.used,memory.total',
 '--format=csv,noheader,nounits'],text=True)
rows=[[int(v.strip()) for v in l.split(',')] for l in gpu.splitlines()]
assert [r[0] for r in rows]==[0,1] and all(r[1]<500 for r in rows)
print(json.dumps(dict(status='FOUR_COMPLETED_ENDPOINTS_VERIFIED_REMAINDER_ABSENT',
 old_exit_sha256=sha(old/'EXIT.json'),gpu_memory_only=rows,
 boundary='Only missingRGBNT100 endpoints may run; old failure and12 completed modes retained.')))
'''
    (LOCAL/'QUALIFY_SOURCE.py').write_text(source)
    client=paramiko.SSHClient()
    client.load_host_keys(str(HOME/'.ssh/known_hosts'))
    client.connect('172.19.12.138',port=2026,username='gaob',key_filename=str(HOME/'.ssh/id_ed25519'),timeout=20)
    stdin,stdout,stderr=client.exec_command("CUDA_VISIBLE_DEVICES='' "+PYTHON+' -B -')
    stdin.write(source);stdin.channel.shutdown_write();stdout.channel.settimeout(120)
    data,error=stdout.read(),stderr.read();code=stdout.channel.recv_exit_status()
    (LOCAL/'QUALIFY_STDOUT.json').write_bytes(data);(LOCAL/'QUALIFY_STDERR.txt').write_bytes(error)
    (LOCAL/'QUALIFY_EXIT.json').write_text(json.dumps(dict(exit_code=code))+'\n')
    assert code==0,error.decode()
    supervisor='settings='+repr(settings)+'\n'+'''
from datetime import datetime
import hashlib,json,os,subprocess
from pathlib import Path
launch=Path(settings['launch']);diag=Path(settings['diagnosis']);root=Path(settings['root'])
def write(name,value):(launch/name).write_text(json.dumps(value,indent=2)+'\\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
steps=[]
for variant in ('semantic','native'):
 command=[settings['python'],'-B',str(root/settings['source']),'--variant',variant,'--diagnosis',str(diag)]
 with (launch/(variant+'.log')).open('x') as log:
  child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
  step=dict(stage='diagnose_'+variant,command=command,pid=child.pid,status='RUNNING',started_at=datetime.now().astimezone().isoformat())
  steps.append(step);write('CHILD.json',dict(status='RUNNING',steps=steps))
  code=child.wait()
 step.update(exit_code=code,status='COMPLETE' if code==0 else 'FAILED',completed_at=datetime.now().astimezone().isoformat())
 write('CHILD.json',dict(status=step['status'],steps=steps))
 if code:
  write('EXIT.json',dict(exit_code=code,steps=steps,completed_at=datetime.now().astimezone().isoformat()));raise SystemExit(code)
 assert sha(root/settings['source'])==settings['source_sha256']
seal=json.loads((diag/'INPUT_SEAL.json').read_text())
matrix=json.loads((Path(seal['original_campaign'])/'accepted_matrix.json').read_text())
rows=[]
for row in matrix['rows']:
 folder=diag/(row['dataset']+'_'+row['variant'])
 endpoint=json.loads((folder/'ENDPOINT.json').read_text())
 assert len(endpoint['modes'])==3
 if row['dataset']=='RGBNT100':
  receipt=json.loads((folder/'FRESH_PROCESS_RECEIPT.json').read_text())
  assert receipt['status']=='FRESH_PROCESS_ENDPOINT_COMPLETE' and receipt['source_sha256']==settings['source_sha256']
  assert receipt['endpoint_sha256']==sha(folder/'ENDPOINT.json')
 rows.append(endpoint)
assert all(sha(root/n)==h for n,h in seal['original_source_sha256'].items())
assert all(sha(root/n)==h for n,h in seal['diagnostic_source_sha256'].items())
summary=dict(schema=seal['schema'],status='COMPLETE',accepted_models=6,deployment_modes=18,
 completed_at=datetime.now().astimezone().isoformat(),rows=rows,
 original_failed_launch=settings['old'],original_failed_exit_sha256=sha(Path(settings['old'])/'EXIT.json'),
 fresh_process_resume_launch=str(launch),resume_source_sha256=settings['source_sha256'],
 boundary='Four prior accepted endpoints plus two missing fresh-process endpoints. Old failure unchanged; no original12mode reruns, training or tolerance changes.')
assert not (diag/'SUMMARY.json').exists()
(diag/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\\n')
command=[settings['python'],'-B',str(root/'tools/report_row_mass_fixed_best.py'),'--diagnosis',str(diag),'--output',settings['report']]
with (launch/'report.log').open('x') as log:
 child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
 step=dict(stage='report',command=command,pid=child.pid,status='RUNNING',started_at=datetime.now().astimezone().isoformat());steps.append(step)
 write('CHILD.json',dict(status='RUNNING',steps=steps));code=child.wait()
step.update(exit_code=code,status='COMPLETE' if code==0 else 'FAILED',completed_at=datetime.now().astimezone().isoformat())
write('CHILD.json',dict(status=step['status'],steps=steps))
write('EXIT.json',dict(exit_code=code,steps=steps,completed_at=datetime.now().astimezone().isoformat()))
raise SystemExit(code)
'''
    (LOCAL/'SUPERVISOR.py').write_text(supervisor)
    sftp=client.open_sftp();sftp.mkdir(LAUNCH)
    sftp.put(str(PACKAGE/NAME),ROOT+'/'+NAME)
    sftp.put(str(LOCAL/'SUPERVISOR.py'),LAUNCH+'/SUPERVISOR.py');sftp.close()
    launcher='settings='+repr(settings)+'\n'+'''
from datetime import datetime
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(settings['root']);folder=Path(settings['launch'])
assert hashlib.sha256((root/settings['source']).read_bytes()).hexdigest()==settings['source_sha256']
assert not (folder/'LAUNCH.json').exists()
with (folder/'console.log').open('x') as log:
 child=subprocess.Popen([settings['python'],'-B',str(folder/'SUPERVISOR.py')],cwd=root,
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,
  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=(Path('/proc')/str(child.pid)/'stat').read_text()
result=dict(status='MISSING_TWO_FIXED_BEST_ENDPOINTS_LAUNCHED',pid=child.pid,
 start_ticks=int(stat.split(') ')[1].split()[19]),at=datetime.now().astimezone().isoformat(),launch=str(folder),settings=settings)
(folder/'LAUNCH.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
'''
    (LOCAL/'LAUNCH_SOURCE.py').write_text(launcher)
    stdin,stdout,stderr=client.exec_command("CUDA_VISIBLE_DEVICES='' "+PYTHON+' -B -')
    stdin.write(launcher);stdin.channel.shutdown_write();stdout.channel.settimeout(60)
    data,error=stdout.read(),stderr.read();code=stdout.channel.recv_exit_status();client.close()
    (LOCAL/'LAUNCH_STDOUT.json').write_bytes(data);(LOCAL/'LAUNCH_STDERR.txt').write_bytes(error)
    (LOCAL/'LAUNCH_EXIT.json').write_text(json.dumps(dict(exit_code=code))+'\n')
    assert code==0,error.decode()
    result=json.loads(data);(LOCAL/'LAUNCH.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
