from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');rows=[]
seal=json.loads((root/'refine-logs/independent_role_heads_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
for folder in sorted((root/'results').iterdir()):
    if folder.is_dir() and 'fixed_best' in folder.name:
        for p in folder.rglob('*.pt'):
            st=p.stat()
            rows.append(dict(path=str(p),bytes=st.st_size,nlink=st.st_nlink,
                current_seal_dependency=str(p) in seal['artifact_sha256']))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),free_bytes=shutil.disk_usage(root).free,
    files=rows,total_bytes=sum(r['bytes'] for r in rows),
    boundary='Read-only inventory of own project fixed-best diagnostic caches. No hashing, removal, NN, scorer replay or other-project action.')))
