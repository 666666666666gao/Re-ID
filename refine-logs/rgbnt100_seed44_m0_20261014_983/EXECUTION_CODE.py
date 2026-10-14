import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

w=Path(__file__).resolve().parent;r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
assert json.loads((w/'SEED44_REGISTRATION982_PUBLICATION.json').read_bytes())['status']=='EXACT_SEED44_REGISTRATION982_SYNC'
assert not (w/'SEED44_RGBNT100_LAUNCH.json').exists()
base='refine-logs/near_signal_rgbnt100_seed44_20261014_982/'
registration=json.loads((r/base/'REGISTRATION.json').read_bytes());pins=registration['source_sha256']
ssh=['ssh','-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-p','2026','gaob@172.19.12.138','/usr/bin/python3','-']
supervisor='pins='+repr(pins)+'\n'+r'''
from datetime import datetime
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');admin=root/'logs/near_signal_rgbnt100_seed44_launch982'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/name)==digest for name,digest in pins.items())
cmd=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/queue_official_extra_seed.py'),
    '--seed','44','--machine','new','--dataset','RGBNT100','--method','R2','--gpu','1',
    '--run-label','near_signal_rgbnt100_seed44_20261014_982','--checkpoint-policy','best_official_map','--epochs','50']
start=time.perf_counter()
with (admin/'stdout').open('wb') as stdout,(admin/'stderr').open('wb') as stderr:
    p=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=stdout,stderr=stderr)
    row=dict(dataset='RGBNT100',seed=44,gpu=1,pid=p.pid,command=cmd,started_at=datetime.now().astimezone().isoformat())
    (admin/'RUNNING.json').write_text(json.dumps(row,indent=2)+'\n');status=p.wait()
row.update(exit_code=status,elapsed_seconds=time.perf_counter()-start,completed_at=datetime.now().astimezone().isoformat(),
    source_seal_unchanged=all(sha(root/name)==digest for name,digest in pins.items()))
(admin/'EXIT.json').write_text(json.dumps(row,indent=2)+'\n');assert row['source_seal_unchanged']
'''
code='pins='+repr(pins)+'\nsupervisor='+repr(supervisor)+'\n'+r'''
from datetime import datetime,timedelta
import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');admin=root/'logs/near_signal_rgbnt100_seed44_launch982'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not admin.exists();assert all(sha(root/name)==digest for name,digest in pins.items())
free=shutil.disk_usage(root).free;assert free>3*1024**3,free
gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
index,memory=map(int,gpu.strip().split(','));assert index==1 and memory<500,(index,memory)
admin.mkdir();(admin/'GPU1_MEMORY_ADMISSION.txt').write_text(gpu)
(admin/'SOURCE_SEAL.json').write_text(json.dumps(pins,indent=2)+'\n')
(admin/'supervisor.py').write_text(supervisor)
with (admin/'supervisor.stdout').open('wb') as stdout,(admin/'supervisor.stderr').open('wb') as stderr:
    p=subprocess.Popen(['/usr/bin/python3',str(admin/'supervisor.py')],cwd=root,stdout=stdout,stderr=stderr,start_new_session=True)
now=datetime.now().astimezone()
launch=dict(status='STARTED_ONCE',dataset='RGBNT100',seed=44,method='R2',epochs=50,gpu=1,port=2026,
    supervisor_pid=p.pid,started_at=now.isoformat(),m0_check_due=(now+timedelta(seconds=300)).isoformat(),
    admin=str(admin),free_disk_bytes=free,registration='refine-logs/near_signal_rgbnt100_seed44_20261014_982/REGISTRATION.json',
    boundary='One additional near-only seed onGPU1. Original seed43GPU0 and observer7051 unchanged. No early original-job metric/GPU/T-P query; canonical503 sources unchanged.')
(admin/'LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch))
'''
run=subprocess.run(ssh,input=code.encode(),capture_output=True)
(w/'SEED44_RGBNT100_LAUNCH.stdout').write_bytes(run.stdout);(w/'SEED44_RGBNT100_LAUNCH.stderr').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode()
launch=json.loads(run.stdout);(w/'SEED44_RGBNT100_LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\n',encoding='utf-8')
print(json.dumps(launch),flush=True)
due=datetime.fromisoformat(launch['m0_check_due']);time.sleep(max(0,(due-datetime.now().astimezone()).total_seconds()))
code=r'''
import base64,hashlib,json
from datetime import datetime,timedelta
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');admin=root/'logs/near_signal_rgbnt100_seed44_launch982'
name='official_extra_seed44_RGBNT100_R2_near_signal_rgbnt100_seed44_20261014_982_bestmap_e50_20260926'
campaign=root/'logs'/name;state=json.loads((campaign/'campaign.json').read_text())
assert state['seed']==44 and state['training_epochs']==50 and len(state['jobs'])==1
receipt_path=campaign/'m0/RGBNT100_R2_seed44/training.json';receipt=json.loads(receipt_path.read_text())
assert receipt['status']=='M0_PASS' and receipt['seed']==44
assert receipt['training']['optimizer_steps']==8 and receipt['training']['frozen_state_unchanged']
assert receipt['training']['missing_nonzero_gradients']==[] and receipt['training']['overflow_events']==0
old_path=root/'trained-model/official_extra_seed42_RGBNT100_R2_horizon50_v1_bestmap_e50_20260926/RGBNT100_R2_seed42/training.json'
old=json.loads(old_path.read_text());assert old['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and old['training']['epochs']==50
elapsed=(datetime.fromisoformat(old['completed_at'])-datetime.fromisoformat(old['started_at'])).total_seconds()
fresh=datetime.fromisoformat(state['jobs'][0]['m0_at']);estimate=fresh+timedelta(seconds=elapsed)
running=json.loads((admin/'RUNNING.json').read_text());assert (Path('/proc')/str(running['pid'])).exists()
result=dict(status='M0_PASS_FRESH_TRAIN_CONTROLLER_LIVE',dataset='RGBNT100',seed=44,state=state,m0=receipt,
    controller=running,fresh_train_start=fresh.isoformat(),old_full50_elapsed_seconds=elapsed,
    old_train_only_seconds=sum(row['seconds'] for row in old['training']['history']),
    estimated_train_complete=estimate.isoformat(),near_finish_due=(estimate-timedelta(minutes=3)).isoformat(),
    observed_at=datetime.now().astimezone().isoformat(),
    boundary='One+300s new-job M0qualification, not an early training epoch/metric poll. Estimate includes original full50 official epoch-evaluation wall time; parallel runtime can vary. Original43observer/queue not queried or restarted.')
files=[receipt_path,campaign/'campaign.json',admin/'RUNNING.json',admin/'LAUNCH.json',admin/'SOURCE_SEAL.json',admin/'stderr']
print(json.dumps(dict(result=result,files={str(p.relative_to(root)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),data=base64.b64encode(p.read_bytes()).decode()) for p in files})))
'''
run=subprocess.run(ssh,input=code.encode(),capture_output=True)
(w/'SEED44_RGBNT100_M0.stdout').write_bytes(run.stdout);(w/'SEED44_RGBNT100_M0.stderr').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode()
value=json.loads(run.stdout);dest=w/'seed44_m0';dest.mkdir()
for name,item in value.pop('files').items():
    data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256']
    path=dest/name;assert path.resolve().is_relative_to(dest.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(w/'SEED44_RGBNT100_M0_OBSERVED.json').write_text(json.dumps(value['result'],indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:value['result'][key] for key in ('status','dataset','seed','fresh_train_start','estimated_train_complete','near_finish_due')}),flush=True)
