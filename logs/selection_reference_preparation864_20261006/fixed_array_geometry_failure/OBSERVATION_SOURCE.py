launch={'status': 'CPU_FIXED_ARRAY_DIAGNOSIS_LAUNCHED', 'pid': 3771483, 'start_ticks': 50867796, 'at': '2026-10-06T10:22:17.196597+08:00', 'launch': '/data/gaob/Re-ID/Trifusion/logs/additive_geometry_fixed_arrays_launch_20261006_864', 'output': '/data/gaob/Re-ID/Trifusion/results/additive_geometry_fixed_arrays_20261006_864', 'no_nn': True, 'no_gpu': True, 'first_observation_due': '2026-10-06T10:25:17.196597+08:00'}
from pathlib import Path
from datetime import datetime
import hashlib,json
folder=Path(launch['launch']);p=Path('/proc')/str(launch['pid'])/'stat'
live=p.is_file() and int(p.read_text().split(') ')[1].split()[19])==launch['start_ticks']
exit_path=folder/'EXIT.json';exit=json.loads(exit_path.read_text()) if exit_path.is_file() else None
root=Path(launch['output']);summary=json.loads((root/'SUMMARY.json').read_text()) if (root/'SUMMARY.json').is_file() else None
files=[]
if exit is not None:
 for directory in (folder,root):
  if directory.exists():
   for path in sorted(directory.rglob('*')):
    if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.txt','.py','.md'):
     files.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 if summary is not None:
  for row in summary['rows']:
   path=root/(row['dataset']+'_'+row['variant'])/'direct_sum_distances.pt'
   assert hashlib.sha256(path.read_bytes()).hexdigest()==row['new_distance_sha256']
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),supervisor_live=live,exit=exit,summary=summary,files=files,
 stderr=(folder/'stderr.txt').read_text() if exit is not None else None,boundary='One CPU saved-array report; no NN/checkpoint/GPU. An unfinished observation does not authorize a restart.')))
