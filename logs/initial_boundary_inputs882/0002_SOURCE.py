from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
scope=json.loads((root/'refine-logs/incremental_role_objective_v1/PENDING_M0_SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(scope)==398 and all(sha(root/n)==h for n,h in scope.items())
controls=json.loads((root/'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json').read_text())
assert len(controls['artifact_sha256'])==45 and all(sha(Path(n))==h for n,h in controls['artifact_sha256'].items())
campaign=root/'logs/incremental_pending_m0_v1_20261007_880'
run=root/'trained-model/incremental_pending_m0_v1_20261007_880_m0_repair_keep_RGBNT201'
receipt=json.loads((run/'training.json').read_text())
assert receipt['status']=='M0_PASS' and receipt['history'][0]['steps']==8
assert json.loads((campaign/'campaign.json').read_text())['formal_eligible'] is False
extra=[run/'training.json',run/'training_batch_order.jsonl',campaign/'initialization/RGBNT201_repair_keep.json']
assert not (run/'m0_reload_probe.pth').exists()
artifacts=dict(controls['artifact_sha256']);artifacts.update({str(p):sha(p) for p in extra});assert len(artifacts)==48
print(json.dumps(dict(status='INITIAL_BOUNDARY_INPUTS_PHYSICALLY_QUALIFIED',at=datetime.now().astimezone().isoformat(),
    artifact_sha256=artifacts,prior_source_count=398,free_bytes=shutil.disk_usage(root).free,
    boundary='Read-only preparation for fresh-public initial one-forward diagnostic. Current45 controls and3 retained M0 metadata; no deleted probe required, no neural execution or old-gate change.')))
