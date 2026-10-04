from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');folder=root/'trained-model'
rows=[{'path':str(p),'bytes':p.stat().st_size} for p in folder.glob('global_task_role_v1_20261004_824*/*.pth')]
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'disk_free_bytes':shutil.disk_usage(root).free,'checkpoint_rows':rows,'checkpoint_bytes':sum(r['bytes'] for r in rows),'boundary':'Read-only own-campaign file metadata, no tensors/model/GPU, no deletion or power/temperature access.'}))
