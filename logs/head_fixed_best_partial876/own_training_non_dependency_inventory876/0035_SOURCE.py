from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');rows=[]
seal=json.loads((root/'refine-logs/independent_role_heads_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
prefixes=('native_research_','role_input_detach_','global_task_role_','deployment_metric_',
    'semantic_capacity_','region_reconstruction_','selective_','role_interaction_','signal_selection_',
    'independent_role_heads_','training_feature_scale_','metric_feature_scale_')
for folder in sorted((root/'trained-model').iterdir()):
    if folder.is_dir() and folder.name.startswith(prefixes):
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ('.pth','.pt'):
                st=p.stat()
                if str(p) not in seal['artifact_sha256']:
                    rows.append(dict(path=str(p),bytes=st.st_size,nlink=st.st_nlink,
                        receipt_exists=(p.parent/'full50_receipt.json').exists(),
                        strict_receipt_exists=(p.parent/'official_eval.json').exists()))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),free_bytes=shutil.disk_usage(root).free,
    files=rows,total_bytes=sum(r['bytes'] for r in rows),
    boundary='Read-only exact own closed-family file inventory. Current seal dependencies excluded; presence outside seal alone does not authorize retirement. No removal/NN/scorer/other-project action.')))
