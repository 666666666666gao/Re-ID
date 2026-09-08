from pathlib import Path
from datetime import datetime
import hashlib,json,subprocess,shlex,time,shutil
p=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
commit='2e947a4325144e37fed638105ac954e7e54b5fe5'
cfg=p/'configs/MSVR310/TriFusion-smooth-ap-paired-v1.json'
digest='974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302'
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4')
session='tri_smooth_ap_2e947a4';log=root.with_suffix('.launch.log')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()==commit
assert hashlib.sha256(cfg.read_bytes()).hexdigest()==digest
assert not root.exists() and not log.exists()
assert shutil.disk_usage(root.parent).free>=4*1024**3
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
assert int(gpu.strip())<500
command=['/root/miniconda3/envs/tri_reid/bin/python','-u','-m','tools.run_msvr_smooth_ap','--config',str(cfg),'--config-sha256',digest,'--code-commit',commit,'--output-dir',str(root)]
shell='cd '+shlex.quote(str(p))+' && exec env PYTHONPATH='+shlex.quote(str(p/'modeling')+':'+str(p))+' '+shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1'
launched=datetime.now().astimezone().isoformat()
subprocess.run(['screen','-dmS',session,'bash','-lc',shell],check=True)
time.sleep(4)
assert (root/'pipeline.json').exists(),log.read_text()
receipt=json.loads((root/'pipeline.json').read_bytes())
print(json.dumps(dict(launched_at=launched,observed_at=datetime.now().astimezone().isoformat(),session=session,command=command,root=str(root),pipeline=receipt,processes={str(pid):Path('/proc',str(pid)).exists() for pid in [receipt['wrapper_pid'],*[s['original_pid'] for s in receipt['stages']]]},free_bytes=shutil.disk_usage(root.parent).free)))
