from pathlib import Path
from datetime import datetime
import json, shutil, subprocess, hashlib
root=Path('/data/gaob/Re-ID/Trifusion')
active=[]
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
        if '/data/gaob/Re-ID/Trifusion/tools/' in cmd: active.append(dict(pid=p.name,command=cmd))
assert not active,active
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
assert head=='7e684e00b0f29f00dd847c14864abf33676c6759'
seal=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(seal['artifact_sha256'])==187 and len(seal['rows'])==9
memory=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
inventory=[]
protected=set(seal['artifact_sha256'])
for p in (root/'trained-model').glob('*/best_map.pth'):
    if str(p) in protected or p.stat().st_size<200*1024**2: continue
    training=p.parent/'training.json'; receipt=p.parent/'official_metrics.json'
    if not training.exists() or not receipt.exists(): continue
    t=json.loads(training.read_text()); r=json.loads(receipt.read_text())
    inventory.append(dict(path=str(p),bytes=p.stat().st_size,nlink=p.stat().st_nlink,dataset=t['dataset'],
                          status=t['status'],epochs=t['epochs'],metrics=r['metrics']))
result=dict(at=datetime.now().astimezone().isoformat(),head=head,own_producers=active,gpu_memory_only=memory,
            free_bytes=shutil.disk_usage(root).free,inventory=inventory,raw_rows=seal['rows'],
            optimizer_source=(root/'comparators/Signal-cd1b0a6/solver/make_optimizer.py').read_text())
print(json.dumps(result))
