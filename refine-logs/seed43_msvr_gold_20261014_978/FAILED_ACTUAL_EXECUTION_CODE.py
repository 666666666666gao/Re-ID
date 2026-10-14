import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

w=Path(__file__).resolve().parent;r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
published=json.loads((w/'SEED43_MSVR_GRID_PUBLICATION.json').read_bytes())
assert published['status']=='EXACT_SEED43_MSVR_GRID_SYNC'
assert not (w/'SEED43_MSVR_ACTUAL_LAUNCH.json').exists()
base='refine-logs/seed43_msvr_budget_grid_20261014_976/'
reg=json.loads((r/base/'ACTUAL_REGISTRATION.json').read_bytes())
pins=reg['source_sha256']
ssh=['ssh','-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-p','2026','gaob@172.19.12.138','/usr/bin/python3','-']
supervisor='pins='+repr(pins)+'\nbase='+repr(base)+'\n'+r'''
from datetime import datetime
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');admin=root/'logs/seed43_msvr_actual976';out=root/'results/seed43_msvr_actual976'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/name)==digest for name,digest in pins.items())
cmd=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/verify_seed43_near_signal_residual_budget.py'),
    '--dataset','MSVR310','--registration',str(root/base/'ACTUAL_REGISTRATION.json'),
    '--comparison',str(root/base/'ACTUAL_COMPARISON.json'),'--output-dir',str(out)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
start=time.perf_counter()
with (admin/'stdout').open('wb') as stdout,(admin/'stderr').open('wb') as stderr:
    p=subprocess.Popen(cmd,cwd=root,env=env,stdout=stdout,stderr=stderr)
    row=dict(dataset='MSVR310',seed=43,gpu=1,pid=p.pid,command=cmd,started_at=datetime.now().astimezone().isoformat())
    (admin/'RUNNING.json').write_text(json.dumps(row,indent=2)+'\n');status=p.wait()
row.update(exit_code=status,elapsed_seconds=time.perf_counter()-start,completed_at=datetime.now().astimezone().isoformat(),source_seal_unchanged=all(sha(root/name)==digest for name,digest in pins.items()))
(admin/'EXIT.json').write_text(json.dumps(row,indent=2)+'\n')
assert row['source_seal_unchanged']
'''
code='pins='+repr(pins)+'\nsupervisor='+repr(supervisor)+'\n'+r'''
from datetime import datetime,timedelta
import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');admin=root/'logs/seed43_msvr_actual976';out=root/'results/seed43_msvr_actual976'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not admin.exists() and not out.exists()
assert all(sha(root/name)==digest for name,digest in pins.items())
original_exit=json.loads((root/'logs/near_signal_seed43_launch970/MSVR310_EXIT.json').read_text());assert original_exit['exit_code']==0
assert shutil.disk_usage(root).free>=591*1055*4+256*1024**2
gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
index,memory=map(int,gpu.strip().split(','));assert index==1 and memory<500
admin.mkdir();(admin/'GPU1_ADMISSION.txt').write_text(gpu)
(admin/'SOURCE_SEAL.json').write_text(json.dumps(pins,indent=2)+'\n')
(admin/'supervisor.py').write_text(supervisor)
with (admin/'supervisor.stdout').open('wb') as stdout,(admin/'supervisor.stderr').open('wb') as stderr:
    p=subprocess.Popen(['/usr/bin/python3',str(admin/'supervisor.py')],cwd=root,stdout=stdout,stderr=stderr,start_new_session=True)
start=datetime.now().astimezone()
receipt=dict(status='STARTED_ONCE',supervisor_pid=p.pid,started_at=start.isoformat(),check_due=(start+timedelta(seconds=180)).isoformat(),
    expected_seconds=120,dataset='MSVR310',seed=43,ratio=.25,gpu=1,port=2026,admin=str(admin),results=str(out),optimizer_updates=0,
    boundary='Only completed queue GPU1 admission checked; RGBNT100 GPU0 and temperature/power untouched. One full frozen encoding, unchanged strict checks, one delayed observer, no retry.')
(admin/'LAUNCH.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''
run=subprocess.run(ssh,input=code.encode(),capture_output=True)
(w/'SEED43_MSVR_ACTUAL_LAUNCH.stdout').write_bytes(run.stdout);(w/'SEED43_MSVR_ACTUAL_LAUNCH.stderr').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode()
launch=json.loads(run.stdout);(w/'SEED43_MSVR_ACTUAL_LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\n',encoding='utf-8')
print(json.dumps(launch),flush=True)
due=datetime.fromisoformat(launch['check_due']);time.sleep(max(0,(due-datetime.now().astimezone()).total_seconds()))
code=r'''
import base64,hashlib,json
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');admin=root/'logs/seed43_msvr_actual976';out=root/'results/seed43_msvr_actual976'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in admin.iterdir() if p.is_file()]
paths=[p for p in out.rglob('*') if p.is_file() and p.name!='official_distances.pt']
files+=paths
terminal=(admin/'EXIT.json').is_file()
record=dict(status='TERMINAL' if terminal else 'RUNNING',at=datetime.now().astimezone().isoformat(),
    exit=json.loads((admin/'EXIT.json').read_text()) if terminal else None,
    summary=json.loads((out/'SUMMARY.json').read_text()) if (out/'SUMMARY.json').is_file() else None)
print(json.dumps(dict(record=record,files={str(p.relative_to(root)):dict(sha256=sha(p),data=base64.b64encode(p.read_bytes()).decode()) for p in files})))
'''
run=subprocess.run(ssh,input=code.encode(),capture_output=True)
(w/'SEED43_MSVR_ACTUAL_OBSERVATION.stdout').write_bytes(run.stdout);(w/'SEED43_MSVR_ACTUAL_OBSERVATION.stderr').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode()
value=json.loads(run.stdout);dest=w/'seed43_msvr_actual';dest.mkdir()
for name,item in value.pop('files').items():
    data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256']
    path=dest/name;assert path.resolve().is_relative_to(dest.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(w/'SEED43_MSVR_ACTUAL_OBSERVED.json').write_text(json.dumps(value['record'],indent=2)+'\n',encoding='utf-8')
record=value['record'];summary=record['summary']
print(json.dumps(dict(status=record['status'],exit=record['exit'],summary=None if summary is None else {key:summary[key] for key in ('status','dataset','seed','ratio','outputs','delta_vs_signal','all_reported_metrics_strictly_above_signal')})),flush=True)
