from pathlib import Path
from datetime import datetime
import json,hashlib
root=Path('/data/gaob/Re-ID/Trifusion');j=root/'logs/incremental_fixed_m0_scale_launch_20261007_879';out=root/'results/incremental_fixed_m0_scale_20261007_879/DIAGNOSIS.json'
assert json.loads((j/'EXIT.json').read_text())['exit_code']==0
child=json.loads((j/'CHILD.json').read_text());assert not (Path('/proc')/str(child['pid'])).exists()
d=json.loads(out.read_text());assert d['status']=='FIXED_M0_SCALE_COMPARISON_COMPLETE' and d['optimizer_updates']==0 and d['model_state_restored_exact']
source=json.loads((root/'refine-logs/incremental_role_objective_v1/FIXED_M0_SCALE_SOURCE_SCOPE.json').read_text())['source_sha256'];assert len(source)==397
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in source.items())
seal=json.loads((root/'refine-logs/incremental_role_objective_v1/FIXED_M0_SCALE_INPUT_SEAL.json').read_text())
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in seal['artifact_sha256'].items())
controls=json.loads((root/'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json').read_text())
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in controls['artifact_sha256'].items())
files=[p for p in j.iterdir() if p.is_file()]+[out]
print(json.dumps(dict(status='COMPLETE_PHYSICAL_SHA_VERIFIED',at=datetime.now().astimezone().isoformat(),
    files={str(p.relative_to(root)):dict(text=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files},
    source_count=397,input_count=9,current45_inputs_unchanged=True,optimizer_updates=0,
    boundary='Original isolated first-eight gate remains FAIL; one post-eight fixed state and source batch only. No original gradient reconstruction or new retrieval result.')))
