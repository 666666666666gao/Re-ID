from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,zipfile

base=Path('/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_signal_source_oof_v1_r2_seed42_e699eac')
target=base/'m0/fold_0/signal_m0.pth'
rows=[]
for p in sorted(base.rglob('*')):
    if p.is_file():
        r=dict(path=str(p),bytes=p.stat().st_size)
        if p.suffix in ('.json','.log','.txt') and p.stat().st_size<100000:
            r['text']=p.read_text()
        rows.append(r)
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),target=str(target),target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),is_zipfile=zipfile.is_zipfile(target),files=rows,free_bytes=shutil.disk_usage(base).free,deleted_files=0)))
