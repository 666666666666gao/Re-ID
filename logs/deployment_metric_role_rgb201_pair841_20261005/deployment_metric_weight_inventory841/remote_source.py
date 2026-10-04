from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
folder=root/'trained-model'
seal=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
protected=set(seal['artifact_sha256'])
rows=[]
for path in sorted(folder.rglob('*.pth')):
 rows.append(dict(path=str(path),bytes=path.stat().st_size,protected_current_control=str(path) in protected,
                  sibling_texts=sorted(p.name for p in path.parent.glob('*.json'))))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),disk_free_bytes=shutil.disk_usage(root).free,
                     weights=rows,total_weight_bytes=sum(r['bytes'] for r in rows),
                     boundary='Own trained-model weight file metadata only,not a deletion eligibility judgment. No model/GPU/progress query or power/temperature action.')))
