from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
seal=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
protected=set(seal['artifact_sha256'])
matrix=json.loads((root/'logs/independent_role_heads_v1_20261006_873/accepted_matrix.json').read_text())
protected.update(str(Path(r['run_dir'])/'best_map.pth') for r in matrix['rows'])
rows=[]
for prefix in ('foundation_recipe_','training_feature_scale_','metric_feature_scale_'):
 for folder in (root/'trained-model').glob(prefix+'*_full_*'):
  p=folder/'best_map.pth'
  if p.is_file():
   receipt=json.loads((folder/'official_metrics.json').read_text())
   rows.append(dict(path=str(p),bytes=p.stat().st_size,nlink=p.stat().st_nlink,
       protected=str(p) in protected,dataset=receipt['dataset'],recipe=receipt['recipe'],
       status=receipt['status'],best_epoch=receipt['selected_epoch'],metrics=receipt['metrics']))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),status='EXISTING_OWN_CLOSED_FOUNDATION_WEIGHT_INVENTORY',
 rows=rows,free_bytes=shutil.disk_usage(root).free,nn_forwards=0,deleted=0,
 boundary='Presence and receipt metadata inventory only, not SHA qualification or permission to delete. Known three own foundation/feature-scale experiment prefixes; no other-project or GPU queries.')))
