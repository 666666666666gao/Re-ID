from pathlib import Path
from datetime import datetime
import json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
seal=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(seal['rows'])==9 and len(seal['artifact_sha256'])==187
caches=[]
for folder in ('results/deployment_metric_fixed_best_five_20261005_850','results/deployment_metric_fixed_best_pending100_20261005_854'):
 for path in (root/folder).glob('*/gallery_features.pt'):
  caches.append(dict(path=str(path),bytes=path.stat().st_size,needed_in_raw_controls=str(path) in seal['artifact_sha256'],diagnosis=(path.parent/'DIAGNOSIS.json').is_file()))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 gpu_memory=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True),
 free_bytes=shutil.disk_usage(root).free,caches=caches,
 processes=subprocess.check_output(['ps','-u','gaob','-o','pid,args'],text=True),
 pending_exit=json.loads((root/'logs/deployment_metric_pending_fixed_best_launch_20261005_854/EXIT.json').read_text()),
 source_scope=json.loads((root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_text()))))
