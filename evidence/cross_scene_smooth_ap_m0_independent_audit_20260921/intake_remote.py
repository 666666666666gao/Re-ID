import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

repo = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
run = Path('/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d')
files = {}
pending = ['configs/MSVR310/TriFusion-cross-scene-smooth-ap-paired-v1.json',
           'protocols/msvr310_train_oof_v1.json',
           'refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_TRACKER.md',
           'evidence/smooth_ap_cross_scene_support_20260909/smooth_ap_cross_scene_support_20260909.json']
bindings = []
while pending:
    name = pending.pop(0)
    if name in files:
        continue
    p = repo / name
    if not p.is_file():
        files[name] = {'exists': False}
        continue
    raw = p.read_bytes()
    files[name] = dict(exists=True, size=len(raw), sha256=hashlib.sha256(raw).hexdigest(), text=raw.decode('utf-8'))
    if p.suffix == '.json':
        value = json.loads(raw)
        if isinstance(value, dict):
            for key in ('project_file_sha256', 'input_bindings'):
                for child, expected in value.get(key, {}).items():
                    if isinstance(expected, str) and not child.startswith('/'):
                        bindings.append(dict(parent=name, child=child, expected=expected))
                        pending.append(child)
            for key in ('previous_config', 'base_config', 'source_config', 'protocol'):
                child = value.get(key)
                if isinstance(child, str) and not child.startswith('/') and (repo / child).is_file():
                    pending.append(child)

artifacts = {}
for p in sorted(run.rglob('*')):
    if not p.is_file() or '/q1/' in str(p):
        continue
    name = str(p.relative_to(run))
    stat = p.stat()
    row = dict(size=stat.st_size, modified_ns=stat.st_mtime_ns)
    if p.suffix in ('.json', '.jsonl', '.log', '.txt') or p.name in ('status', 'exit_code'):
        raw = p.read_bytes()
        row.update(sha256=hashlib.sha256(raw).hexdigest(), text=raw.decode('utf-8'))
    artifacts[name] = row
print(json.dumps(dict(collected_at=datetime.now(timezone.utc).isoformat(), repo=str(repo), run=str(run),
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
    files=files, bindings=bindings, artifacts=artifacts), ensure_ascii=False))
