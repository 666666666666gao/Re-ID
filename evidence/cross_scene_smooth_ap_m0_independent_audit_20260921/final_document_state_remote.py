import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
path=repo/'refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_TRACKER.md'
raw=path.read_bytes()
print(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
    changed_since_execution=subprocess.check_output(['git','diff','--name-only','d35864d6411591e05c8ac3e5164ebae48063ad99','HEAD'],cwd=repo,text=True).splitlines(),
    tracker=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode('utf-8'))),ensure_ascii=False))
