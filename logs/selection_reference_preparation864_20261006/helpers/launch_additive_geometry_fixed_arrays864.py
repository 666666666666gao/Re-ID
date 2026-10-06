from pathlib import Path
from datetime import datetime,timedelta
import ast
import hashlib
import json
import paramiko

base=Path('C:/Users/gb/.codex_tmp')
packet=base/'independent_evidence_draft/additive_geometry_fixed_arrays864'
assert not (packet/'LAUNCH.json').exists()
source=(packet/'diagnose.py').read_text(encoding='utf-8')
ast.parse(source)
fixed=json.loads((base/'fb_complete863/remote_stdout.json').read_bytes())
assert fixed['exit']['exit_code']==0
root='/data/gaob/Re-ID/Trifusion'
launch=root+'/logs/additive_geometry_fixed_arrays_launch_20261006_864'
output=root+'/results/additive_geometry_fixed_arrays_20261006_864'
nnroot=root+'/logs/fixed_best_row_transport_diagnosis_20261006_862'
rows=[]
inputs={}
for row in fixed['nn_summary']['rows']:
    original=next(m for m in row['modes'] if m['mode']=='original')
    folder=nnroot+'/'+row['dataset']+'_'+row['variant']+'/original'
    rows.append(dict(dataset=row['dataset'],variant=row['variant'],best_epoch=row['best_epoch'],
                     gain=row['weight_statistics']['readout_gain'],original_folder=folder))
    inputs[folder+'/diagnostic_arrays.pt']=original['arrays_sha256']
    inputs[folder+'/distances.pt']=original['distance_sha256']
known_sources={
    root+'/tools/report_row_mass_fixed_best.py':'56d10437b598faceb07268e5362e560af91f8c5d5a9a9bb7e21fedcc422e3e8d',
    root+'/tools/run_official_three_dataset_roles.py':None,
    root+'/tools/train_rgbnt100_signal_oof.py':None,
    root+'/tools/train_msvr310_signal_oof.py':None}
formal=json.loads((base/'rt_complete862/remote_stdout.json').read_bytes())
for name in known_sources:
    if known_sources[name] is None:known_sources[name]=formal['source_sha256'][name[len(root)+1:]]
sources={**known_sources,launch+'/diagnose.py':hashlib.sha256(source.encode()).hexdigest()}
settings=dict(schema='fixed-array-additive-geometry-v1',rows=rows,input_sha256=inputs,source_sha256=sources)
supervisor='''from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys
folder=Path(__file__).resolve().parent
settings=json.loads((folder/'SETTINGS.json').read_text())
environment={**os.environ,'CUDA_VISIBLE_DEVICES':''}
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(folder/'diagnose.py'),'--seal',str(folder/'INPUT_SEAL.json'),'--output',settings['output']]
with (folder/'stdout.txt').open('wb') as out,(folder/'stderr.txt').open('wb') as err:
 child=subprocess.Popen(command,cwd='/data/gaob/Re-ID/Trifusion',env=environment,stdout=out,stderr=err)
 record=dict(pid=child.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
 (folder/'CHILD.json').write_text(json.dumps(record,indent=2)+'\n')
 exit_code=child.wait()
record.update(exit_code=exit_code,completed_at=datetime.now().astimezone().isoformat())
(folder/'EXIT.json').write_text(json.dumps(record,indent=2)+'\n')
sys.exit(exit_code)
'''
ast.parse(supervisor)
(packet/'SUPERVISOR.py').write_text(supervisor,encoding='utf-8')
qualification='inputs='+repr(inputs)+'\nsources='+repr(known_sources)+'\nlaunch='+repr(launch)+'\noutput='+repr(output)+'\n'+'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
assert not Path(launch).exists() and not Path(output).exists()
assert json.loads((root/'logs/fixed_best_row_transport_resume_20261006_863/EXIT.json').read_text())['exit_code']==0
active=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit() and (p/'cmdline').is_file():
  cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if '/data/gaob/Re-ID/Trifusion/tools/' in cmd:active.append(p.name)
assert not active,active
assert all(sha(p)==d for p,d in inputs.items()) and all(sha(p)==d for p,d in sources.items())
for p in inputs:
 if p.endswith('distances.pt'):
  q=str(Path(p).parent/'own_global.json');inputs[q]=sha(q)
assert shutil.disk_usage(root).free>2*1024**3+256*1024**2
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),input_sha256=inputs,source_sha256=sources,free_bytes=shutil.disk_usage(root).free,no_nn=True)))
'''
# Extend the existing metadata inputs after iterating a snapshot, not a live dict.
qualification=qualification.replace('for p in inputs:\n','for p in list(inputs):\n')
compile(qualification,'qualification864','exec')
(packet/'QUALIFICATION_SOURCE.py').write_text(qualification,encoding='utf-8')
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
def run(code):
    i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(180)
    data,error=o.read(),e.read();status=o.channel.recv_exit_status()
    assert status==0,error.decode()
    return json.loads(data)
ready=run(qualification)
(packet/'QUALIFICATION.json').write_text(json.dumps(ready,indent=2)+'\n')
settings['input_sha256']=ready['input_sha256']
settings['registered_at']=datetime.now().astimezone().isoformat()
(packet/'INPUT_SEAL.json').write_text(json.dumps(settings,indent=2)+'\n')
run('from pathlib import Path;import json;p=Path('+repr(launch)+');assert not p.exists();p.mkdir();print(json.dumps({"created":str(p)}))')
s=c.open_sftp()
for local,name in ((packet/'diagnose.py','diagnose.py'),(packet/'SUPERVISOR.py','supervisor.py'),(packet/'INPUT_SEAL.json','INPUT_SEAL.json'),(packet/'PLAN.md','PLAN.md')):
    s.put(str(local),launch+'/'+name)
with s.open(launch+'/SETTINGS.json','w') as stream:stream.write(json.dumps(dict(output=output)))
s.close()
started=run('launch='+repr(launch)+'\n'+'''from pathlib import Path
from datetime import datetime
import hashlib,json,subprocess,os
p=Path(launch)
with (p/'supervisor.stdout.txt').open('wb') as out,(p/'supervisor.stderr.txt').open('wb') as err:
 child=subprocess.Popen(['/usr/bin/python3','-B',str(p/'supervisor.py')],cwd='/data/gaob/Re-ID/Trifusion',stdout=out,stderr=err,start_new_session=True)
ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split(') ')[1].split()[19])
record=dict(status='CPU_FIXED_ARRAY_DIAGNOSIS_LAUNCHED',pid=child.pid,start_ticks=ticks,at=datetime.now().astimezone().isoformat(),launch=str(p),output=json.loads((p/'SETTINGS.json').read_text())['output'],no_nn=True,no_gpu=True)
(p/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
''')
c.close()
started['first_observation_due']=(datetime.fromisoformat(started['at'])+timedelta(seconds=180)).isoformat()
(packet/'LAUNCH.json').write_text(json.dumps(started,indent=2)+'\n')
print(json.dumps(started,indent=2))
