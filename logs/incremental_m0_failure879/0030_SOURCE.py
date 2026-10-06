from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');c=root/'logs/incremental_role_objective_m0_v1_20261007_878';j=root/'logs/incremental_role_objective_m0_launch_20261007_878'
exit=json.loads((j/'EXIT.json').read_text());assert exit['exit_code']==1
child=json.loads((j/'CHILD.json').read_text());assert not (Path('/proc')/str(child['pid'])).exists()
source=json.loads((root/'refine-logs/incremental_role_objective_v1/SOURCE_SCOPE.json').read_text())['source_sha256'];assert len(source)==393
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==s for n,s in source.items())
controls=json.loads((root/'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json').read_text())
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==s for n,s in controls['artifact_sha256'].items())
run=root/'trained-model/incremental_role_objective_m0_v1_20261007_878_m0_md_batch_ratio_RGBNT201'
training=json.loads((run/'training.json').read_text());assert training['status']=='M0_PASS'
probe=run/'m0_reload_probe.pth';assert hashlib.sha256(probe.read_bytes()).hexdigest()==training['m0']['reload_probe_sha256']
selected=[run/n for n in ('training.json','training_steps.jsonl','training_batch_order.jsonl','m0_reload_probe.pth')]
selected+=[c/'initialization/RGBNT201_md_batch_ratio.json',j/'EXIT.json',j/'CHILD.json',j/'controller.log',c/'campaign.json']
files=[p for folder in (c,j,run) for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.log','.py')]
value=dict(status='FIXED_FAILED_M0_DIAGNOSTIC_INPUTS_QUALIFIED',at=datetime.now().astimezone().isoformat(),
    original_exit=exit,original_controller_pid=child['pid'],original_training_status=training['status'],
    artifact_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in selected},
    original_text_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    old393_sources_and_current45_controls_unchanged=True,free_bytes=shutil.disk_usage(root).free,
    boundary='Original isolated gate FAIL preserved. Retained first eight-step M0 probe is a current diagnostic input, not a formal model. No model/GPU query or loss/threshold modification.')
print(json.dumps(value))
