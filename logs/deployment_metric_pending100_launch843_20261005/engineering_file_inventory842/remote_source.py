from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');protect=set(json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256'])
rows=[dict(path=str(p),bytes=p.stat().st_size,protected_current=str(p) in protect) for folder in (root/'logs',root/'results') for p in folder.rglob('*') if p.is_file() and p.stat().st_size>=128*1024**2]
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),disk_free_bytes=shutil.disk_usage(root).free,rows=sorted(rows,key=lambda r:r['bytes'],reverse=True),boundary='Own logs/results filesystem metadata only,no deletion eligibility claim or model/power-temperature query.')))
