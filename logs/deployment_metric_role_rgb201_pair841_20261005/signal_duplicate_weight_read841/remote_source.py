from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');folder=root/'trained-model'
paths=['/data/gaob/Re-ID/Trifusion/trained-model/signal_full_amp_audit_20260926/RGBNT201/Signalbest.pth', '/data/gaob/Re-ID/Trifusion/trained-model/signal_full_author_core_20260926/RGBNT201/Signalbest.pth', '/data/gaob/Re-ID/Trifusion/trained-model/signal_full_best_map_20260926/MSVR310/Signalbest.pth', '/data/gaob/Re-ID/Trifusion/trained-model/signal_full_best_map_20260926/RGBNT100/Signalbest.pth', '/data/gaob/Re-ID/Trifusion/trained-model/signal_full_best_map_20260926/RGBNT201/Signalbest.pth', '/data/gaob/Re-ID/Trifusion/trained-model/signal_full_matched_20260925/MSVR310/Signalbest.pth', '/data/gaob/Re-ID/Trifusion/trained-model/signal_full_matched_20260925/RGBNT100/Signalbest.pth', '/data/gaob/Re-ID/Trifusion/trained-model/signal_full_matched_20260925/RGBNT201/Signalbest.pth', '/data/gaob/Re-ID/Trifusion/trained-model/signal_plain_baseline_20260924_r2/MSVR310/Signalbest.pth', '/data/gaob/Re-ID/Trifusion/trained-model/signal_plain_baseline_20260924_r2/RGBNT100/Signalbest.pth', '/data/gaob/Re-ID/Trifusion/trained-model/signal_plain_baseline_20260924_r2/RGBNT201/Signalbest.pth']
rows=[]
for name in paths:
 path=Path(name);assert path.resolve().is_relative_to(folder.resolve())
 stat=path.stat();digest=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
 rows.append(dict(path=name,bytes=stat.st_size,sha256=digest.hexdigest(),device=stat.st_dev,inode=stat.st_ino,links=stat.st_nlink))
groups={}
for row in rows:groups.setdefault(row['sha256'],[]).append(row)
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),disk_free_bytes=shutil.disk_usage(root).free,rows=rows,
                     duplicate_groups=[v for v in groups.values() if len(v)>1],
                     boundary='Observed historical immutable Signal weight copies only. No deletion,NN,GPU,power/temp or running-queue query.')))
