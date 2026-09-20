from pathlib import Path
import datetime, json, shutil, subprocess
p=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
root=Path('/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d')
j=json.loads((root/'pipeline.json').read_bytes())
print(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),
    status=subprocess.check_output(['git','status','--short'],cwd=p,text=True),
    previous_status=j['status'],
    original_processes={str(x):Path('/proc',str(x)).exists() for x in [j['wrapper_pid']]+[s['original_pid'] for s in j['stages']]},
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True),
    output_free_bytes=shutil.disk_usage(root.parent).free,repository_free_bytes=shutil.disk_usage(p).free)))
