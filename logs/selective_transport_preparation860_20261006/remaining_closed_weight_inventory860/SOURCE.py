from pathlib import Path
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
protected=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
rows=[]
for weight in sorted((root/'trained-model').glob('*/best_map.pth')):
 if str(weight) not in protected:
  folder=weight.parent
  train=folder/'training.json';receipt=folder/'official_metrics.json'
  if train.is_file() and receipt.is_file():
   t=json.loads(train.read_text());r=json.loads(receipt.read_text())
   rows.append(dict(directory=str(folder),bytes=weight.stat().st_size,
    epochs=len(t['history']),status=r['status'],metrics=r.get('metrics'),best_epoch=t.get('best_epoch')))
print(json.dumps(dict(status='READ_ONLY_REMAINING_WEIGHT_INVENTORY',free_bytes=shutil.disk_usage(root).free,rows=rows)))