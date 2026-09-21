from pathlib import Path
import json,shutil,subprocess
root=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639')
groups={}
for stage in ('m0','q1'):
    rows={}
    for p in (root/stage).rglob('*'):
        if p.is_file():
            suffix=p.suffix
            cell=rows.setdefault(suffix,dict(files=0,bytes=0))
            cell['files']+=1;cell['bytes']+=p.stat().st_size
    groups[stage]=rows
print(json.dumps(dict(previous_run=str(root),sizes=groups,
    output_free_bytes=shutil.disk_usage(root).free,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True),
    compute_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True),
    changed_files=0),indent=2))
