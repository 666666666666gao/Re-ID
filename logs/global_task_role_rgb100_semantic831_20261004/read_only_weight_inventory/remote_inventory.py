from pathlib import Path
from datetime import datetime
import json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
folder=root/'trained-model'
names=sorted(str(path) for path in folder.rglob('*.pth'))
rows=[{'path':name,'bytes':Path(name).stat().st_size} for name in names]
sizes=subprocess.check_output(['du','-sx','--block-size=1',str(folder),str(root/'logs'),str(root/'results'),str(root/'.git')],text=True)
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'disk_free_bytes':shutil.disk_usage(root).free,'checkpoints':rows,'directory_bytes':sizes,'boundary':'Read-only own trained-model checkpoint metadata and own project directory sizes. No model/GPU/temperature/power access, deletion or current-source modification.'}))
