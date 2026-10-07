from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and (proc/'cmdline').is_file():
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in (proc/'cmdline').read_bytes().split(bytes([0])))
names=('global_task_role_v1_20261004_824','semantic_capacity_control_v1_20261005_856','region_reconstruction','row_mass','row_soft','role_transport','selective_role_transport','incremental_role_objective_m0')
rows=[]
for folder in (root/'trained-model').iterdir():
    if folder.is_dir() and folder.name.startswith(names):
        files=[dict(name=p.name,bytes=p.stat().st_size) for p in folder.iterdir() if p.is_file() and p.suffix in ('.pth','.pt')]
        rows.append(dict(folder=str(folder),files=files))
print(json.dumps(dict(status='READ_ONLY_OWN_CLOSED_FAMILY_INVENTORY',at=datetime.now().astimezone().isoformat(),free_bytes=shutil.disk_usage(root).free,rows=rows)))
