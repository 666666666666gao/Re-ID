import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
run = Path('/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d')
files = {}
paths = [run/n for n in ('pipeline.json','t0.json','t0.log','m0.log','m0_cpu.json','m0_cpu.log')]
paths += sorted(p for p in (run/'m0').rglob('*') if p.is_file())
for p in paths:
    row = dict(size=p.stat().st_size, modified_ns=p.stat().st_mtime_ns)
    if p.suffix in ('.json','.jsonl','.log'):
        raw = p.read_bytes()
        row.update(sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode('utf-8'))
    files[str(p.relative_to(run))] = row
print(json.dumps(dict(collected_at=datetime.now(timezone.utc).isoformat(),files=files),ensure_ascii=False))
