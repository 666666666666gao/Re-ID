import base64
import hashlib
import json
from pathlib import Path
import subprocess

w=Path(__file__).resolve().parent;r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
assert not (w/'SEED43_MSVR_PRECISION.json').exists()
pins=json.loads((w/'seed43_msvr_actual/logs/seed43_msvr_actual976/SOURCE_SEAL.json').read_bytes())
pins['tools/diagnose_msvr_cached_rank_precision.py']=hashlib.sha256((r/'tools/diagnose_msvr_cached_rank_precision.py').read_bytes()).hexdigest()
code='pins='+repr(pins)+'\n'+r'''
import base64,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');py='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
admin=root/'logs/seed43_msvr_precision977';out=root/'results/seed43_msvr_precision977'
assert not admin.exists() and not out.exists();admin.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/name)==digest for name,digest in pins.items())
old=json.loads((root/'logs/seed43_msvr_actual976/EXIT.json').read_text())
assert old['exit_code']==1 and old['seed']==43
assert not (Path('/proc')/str(old['pid'])).exists()
base=root/'refine-logs/seed43_msvr_budget_grid_20261014_976'
cmd=[py,'-B',str(root/'tools/diagnose_msvr_cached_rank_precision.py'),
    '--registration',str(base/'ACTUAL_REGISTRATION.json'),'--comparison',str(base/'ACTUAL_COMPARISON.json'),
    '--failed-run',str(root/'results/seed43_msvr_actual976'),'--output-dir',str(out)]
start=time.perf_counter()
with (admin/'stdout').open('wb') as stdout,(admin/'stderr').open('wb') as stderr:
    p=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=stdout,stderr=stderr);status=p.wait()
receipt=dict(exit_code=status,pid=p.pid,elapsed_seconds=time.perf_counter()-start,command=cmd,
    nn_calls=0,optimizer_updates=0,gpu_queries=0,other_running_dataset_queries=0,
    boundary='This execution diagnoses seed43 actual976; the unchanged historical tool boundary sentence names actual966, which is its earlier provenance only. The explicit failed-run path and rank text hashes identify this execution. No new inference or parameter selection.')
(admin/'EXIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert status==0,(admin/'stderr').read_text()
assert all(sha(root/name)==digest for name,digest in pins.items())
files=[out/'SUMMARY.json',admin/'EXIT.json']
print(json.dumps(dict(receipt=receipt,files={str(p.relative_to(root)):dict(sha256=sha(p),data=base64.b64encode(p.read_bytes()).decode()) for p in files})))
'''
run=subprocess.run(['ssh','-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code.encode(),capture_output=True)
(w/'SEED43_MSVR_PRECISION.stdout').write_bytes(run.stdout);(w/'SEED43_MSVR_PRECISION.stderr').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode()
value=json.loads(run.stdout);dest=w/'seed43_msvr_precision';dest.mkdir()
for name,item in value.pop('files').items():
    data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256']
    path=dest/name;assert path.resolve().is_relative_to(dest.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(w/'SEED43_MSVR_PRECISION.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(execution=value['receipt'],diagnosis=json.loads((dest/'results/seed43_msvr_precision977/SUMMARY.json').read_bytes()))))
