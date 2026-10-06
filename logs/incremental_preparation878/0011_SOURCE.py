from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()
old=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(old['artifact_sha256'])==187
assert all(sha(p)==s for p,s in old['artifact_sha256'].items())
rows=[r for r in old['rows'] if r['variant'] in ('semantic','global_only')]
assert len(rows)==6
prefixes=[r[k]+'/' for r in rows for k in ('run_dir','m0_dir')]
selected={p:s for p,s in old['artifact_sha256'].items() if any(p.startswith(prefix) for prefix in prefixes)}
for d in ('RGBNT201','MSVR310','RGBNT100'):
    p=root/'logs/global_task_role_v1_20261004_824/initialization'/f'{d}_semantic.json'
    value=json.loads(p.read_text());r=next(r for r in rows if r['dataset']==d and r['variant']=='semantic')
    assert value['binding']==r['initializer'];selected[str(p)]=sha(p)
    old_order=root/f'trained-model/global_task_role_v1_20261004_824_m0_semantic_{d}/training_batch_order.jsonl'
    assert len(old_order.read_text().splitlines())>=8
    selected[str(old_order)]=sha(old_order)
seal=dict(schema='trifusion-incremental-role-objective-controls-v1',registered_at=datetime.now().astimezone().isoformat(),
    rows=rows,artifact_sha256=selected,
    boundary='Six unchanged existing raw-semantic/global-only controls. New M0 objective probes and new full50 are not included. Old187 seal remains physically intact at registration.')
gpu=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or not (p/'cmdline').exists():continue
    raw=(p/'cmdline').read_bytes()
    if not raw:continue
    words=raw.split(b'\0')
    if any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/run_') or w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/diagnose_') for w in words):
        processes.append(dict(pid=int(p.name),argv=[w.decode() for w in words if w]))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),control_seal=seal,
    old187_all_physical_sha_valid=True,control_artifacts=len(selected),free_bytes=shutil.disk_usage(root).free,
    physical_01_memory=gpu,own_neural_processes=processes,
    boundary='Readonly old evidence/source receipt and selected GPU0/1 memory only. No GPU2/3,25,power-temperature or model action.')))
