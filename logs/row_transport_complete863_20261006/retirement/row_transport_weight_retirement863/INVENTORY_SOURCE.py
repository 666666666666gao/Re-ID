from pathlib import Path
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
rows=[]
for weight in sorted((root/'trained-model').glob('*/best_map.pth')):
 folder=weight.parent
 if (folder/'training.json').is_file() and (folder/'official_metrics.json').is_file():
  t=json.loads((folder/'training.json').read_text());r=json.loads((folder/'official_metrics.json').read_text())
  rows.append(dict(directory=str(folder),epochs=len(t['history']),status=r['status'],best_epoch=t.get('best_epoch'),metrics=r.get('metrics'),checkpoint_sha256=r.get('checkpoint_sha256'),bytes=weight.stat().st_size))
protected=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
print(json.dumps(dict(at=__import__('datetime').datetime.now().astimezone().isoformat(),free_bytes=shutil.disk_usage(root).free,rows=rows,protected_weights={n:d for n,d in protected.items() if n.endswith('.pth')})))
