from pathlib import Path
from datetime import datetime
import json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion');folder=root/'trained-model'
rows=[{'path':str(p),'bytes':p.stat().st_size} for d in folder.glob('global_task_role_v1_20261004_824*') for p in d.rglob('*') if p.is_file() and p.stat().st_size>10485760]
sizes=subprocess.check_output(['du','-sx','--block-size=1',str(folder),str(root/'logs'),str(root/'results'),str(root/'.git')],text=True)
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'disk_free_bytes':shutil.disk_usage(root).free,'own_large_artifacts':rows,'directory_bytes':sizes,'boundary':'Read-only metadata for current artifacts and own project directories; no tensor/model/GPU, no delete or power/temperature access.'}))
