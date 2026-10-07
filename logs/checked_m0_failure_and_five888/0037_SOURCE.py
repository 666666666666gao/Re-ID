from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');rows=[]
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and (proc/'cmdline').is_file():
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in (proc/'cmdline').read_bytes().split(bytes([0])))
for folder in (root/'trained-model').iterdir():
    if folder.is_dir():
        for p in folder.glob('*.pth'):
            if p.is_file():rows.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size))
print(json.dumps(dict(status='READONLY_PTH_PATH_SIZE_OWN_TRAINED_MODEL_ONLY',at=datetime.now().astimezone().isoformat(),free_bytes=shutil.disk_usage(root).free,rows=sorted(rows,key=lambda r:r['path']))))
