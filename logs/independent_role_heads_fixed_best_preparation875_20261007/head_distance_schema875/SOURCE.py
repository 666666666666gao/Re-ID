from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
import torch
root=Path('/data/gaob/Re-ID/Trifusion')
c=root/'logs/independent_role_heads_v1_20261006_873'
j=root/'logs/independent_role_heads_launch_20261006_873'
s=json.loads((c/'campaign.json').read_text())
assert s['status']=='COMPLETE' and s['report_invocations']==1 and s['report_exit_code']==0
assert json.loads((j/'EXIT.json').read_text())['exit_code']==0
assert not (Path('/proc')/str(s['controller_pid'])).exists()
matrix=json.loads((c/'accepted_matrix.json').read_text())
rows=[]
for r in matrix['rows']:
 p=Path(r['run_dir'])/'official_distances.pt'
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 assert h.hexdigest()==r['distance_sha256']
 data=torch.load(p,map_location='cpu',weights_only=False)
 rows.append(dict(dataset=r['dataset'],path=str(p),bytes=p.stat().st_size,
     sha256=h.hexdigest(),keys=list(data),arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in data.items()}))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),status='CPU_SAVED_DISTANCE_SCHEMA_VERIFIED',rows=rows,
 free_bytes=shutil.disk_usage(root).free,nn_forwards=0,optimizer_updates=0,
 boundary='CPU reads of fixed accepted distance artifacts only. No global/correction feature reconstruction, model, GPU query, scoring replay, source update or deletion.')))
