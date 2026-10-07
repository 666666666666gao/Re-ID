from pathlib import Path
from datetime import datetime
import json,hashlib
root=Path('/data/gaob/Re-ID/Trifusion');j=root/'logs/initial_incremental_boundary_launch_20261007_882';out=root/'results/initial_incremental_boundary_20261007_882/DIAGNOSIS.json'
assert json.loads((j/'EXIT.json').read_text())['exit_code']==0
child=json.loads((j/'CHILD.json').read_text());assert not (Path('/proc')/str(child['pid'])).exists()
d=json.loads(out.read_text());assert d['status']=='INITIAL_AUXILIARY_BOUNDARY_COMPARISON_COMPLETE' and d['optimizer_updates']==0 and d['model_state_restored_exact']
source=json.loads((root/'refine-logs/incremental_role_objective_v1/INITIAL_BOUNDARY_SOURCE_SCOPE.json').read_text())['source_sha256'];assert len(source)==402
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in source.items())
seal=json.loads((root/'refine-logs/incremental_role_objective_v1/INITIAL_BOUNDARY_INPUT_SEAL.json').read_text())
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in seal['artifact_sha256'].items())
controls=json.loads((root/'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json').read_text())
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in controls['artifact_sha256'].items())
files=[p for p in j.iterdir() if p.is_file()]+[out]
print(json.dumps(dict(status='COMPLETE_PHYSICAL_SHA_VERIFIED',at=datetime.now().astimezone().isoformat(),
    files={str(p.relative_to(root)):dict(text=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files},
    source_count=402,input_count=48,current45_inputs_unchanged=True,optimizer_updates=0,
    boundary='Originalsix isolatedfirst-eightgates FAIL unchanged. One fresh-public initial201 AMP graph,one sourcebatch,twoVJP andzero updates only; no original augmentation reconstruction, full qualification or retrieval result.')))
