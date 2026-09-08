from pathlib import Path
from datetime import datetime
import json, shutil

roots=[Path('/root/autodl-tmp/trifusion-v2'),Path('/root/trifusion-storage/artifacts')]
rows=[]
for root in roots:
    weights=[]
    for p in root.rglob('*'):
        if p.is_file() and p.suffix in ('.pth','.pt','.ckpt'):
            st=p.stat()
            weights.append(dict(path=str(p),bytes=st.st_size,symlink=p.is_symlink(),resume='.resume' in p.parts))
    rows.append(dict(root=str(root),free_bytes=shutil.disk_usage(root).free,weights=weights,total_weight_bytes=sum(x['bytes'] for x in weights)))
run=Path('/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739')
pipe=json.loads((run/'pipeline.json').read_bytes())
pids=[pipe['wrapper_pid'],*[s['original_pid'] for s in pipe['stages']]]
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),inventories=rows,original_pid_exists={str(p):Path('/proc',str(p)).exists() for p in pids},pipeline_status=pipe['status'],deleted_files=0)))
