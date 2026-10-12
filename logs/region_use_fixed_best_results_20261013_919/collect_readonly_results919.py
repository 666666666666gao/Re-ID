import hashlib
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

work = Path(__file__).resolve().parent
target = work/'readonly_results919'
assert not target.exists()
remote = r'''
import hashlib,json,zipfile
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion')
out=root/'logs/region_use_fixed_best_diagnosis_20261013_918'
admin=root/'logs/region_use_fixed_best_diagnosis_admin918'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
state=json.loads((out/'campaign.json').read_text())
terminal=json.loads((admin/'EXIT.json').read_text())
assert state['status']=='COMPLETE' and terminal['exit_code']==0
assert len(state['jobs'])==6 and all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
assert all(not Path('/proc/'+str(p)).exists() for p in [state['pid'],terminal['supervisor_pid']]+[j['pid'] for j in state['jobs']])
seal=json.loads((root/'refine-logs/region_use_fixed_best_diagnosis918/INPUT_SEAL.json').read_text())
files=[out/'campaign.json',admin/'EXIT.json',admin/'controller.log',admin/'PUBLICATION_SYNC_918.json']
for job in state['jobs']:
 p=Path(job['output_dir'])/'DIAGNOSIS.json'
 assert sha(p)==job['diagnosis_sha256']
 d=json.loads(p.read_text())
 assert d['status']=='COMPLETE' and d['optimizer_updates']==0
 binary=p.parent/'diagnostic_distances.pt'
 assert binary.stat().st_size==d['distance_artifact']['bytes'] and sha(binary)==d['distance_artifact']['sha256']
 files.append(p)
for path,h in seal['artifact_sha256'].items():
 if 'full_global_only_' in path and path.endswith('training.json'):
  p=Path(path);assert sha(p)==h;files.append(p)
assert sum('full_global_only_' in str(p) for p in files)==3
manifest=dict(at=datetime.now().astimezone().isoformat(),file_sha256={str(p.relative_to(root)):sha(p) for p in files},
 pinned_distances={str(Path(j['output_dir'])/'diagnostic_distances.pt'):json.loads((Path(j['output_dir'])/'DIAGNOSIS.json').read_text())['distance_artifact'] for j in state['jobs']},
 neural_calls_by_collection=0,optimizer_updates=0,original_processes_absent=True)
archive=admin/'READONLY_RESULTS_919.zip'
assert not archive.exists()
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,str(p.relative_to(root)))
 z.writestr('INTAKE_MANIFEST_919.json',json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),campaign_status=state['status'],exit_code=terminal['exit_code'])))
'''
options=['-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-i','C:/Users/gb/.ssh/id_ed25519']
command=['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-']
r=subprocess.run(command,input=remote,text=True,capture_output=True)
(work/'READONLY_COLLECTION_919.stdout').write_text(r.stdout,encoding='utf-8')
(work/'READONLY_COLLECTION_919.stderr').write_text(r.stderr,encoding='utf-8')
(work/'READONLY_COLLECTION_COMMAND_919.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),exit_code=r.returncode,command=command),indent=2)+'\n',encoding='utf-8')
assert r.returncode==0,(r.returncode,r.stderr)
meta=json.loads(r.stdout)
archive=work/'READONLY_RESULTS_919.zip'
assert not archive.exists()
subprocess.run(['scp',*options,'-P','2026','gaob@172.19.12.138:'+meta['archive'],str(archive)],check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(archive)==meta['sha256'] and archive.stat().st_size==meta['bytes']
with zipfile.ZipFile(archive) as z:
 assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
 z.extractall(target)
manifest=json.loads((target/'INTAKE_MANIFEST_919.json').read_text(encoding='utf-8'))
assert all(sha(target/name)==h for name,h in manifest['file_sha256'].items())
(work/'READONLY_INTAKE_919.json').write_text(json.dumps(dict(status='EXACT_TERMINAL_INTAKE',at=datetime.now().astimezone().isoformat(),archive=meta,file_sha256=manifest['file_sha256'],neural_calls=0),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='EXACT_TERMINAL_INTAKE',files=len(manifest['file_sha256']),archive_bytes=meta['bytes'])))
