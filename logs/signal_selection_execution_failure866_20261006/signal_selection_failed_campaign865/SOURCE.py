from pathlib import Path
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/signal_selection_reference_v1_20261006_865'
launch=root/'logs/signal_selection_reference_launch_20261006_865'
assert json.loads((launch/'EXIT.json').read_text())['exit_code']==1
state=json.loads((campaign/'campaign.json').read_text())
assert len(state['preparation'])==3 and all(s['exit_code']==0 for s in state['preparation'])
assert state['failed_job']['phase']=='m0' and state['report_invocations']==0
for folder in (root/'trained-model').glob('signal_selection_reference_v1_20261006_865_*'):assert not folder.exists()
files=[p for folder in (campaign,launch) for p in folder.rglob('*') if p.is_file()]
assert all(p.suffix in ('.json','.log','.txt','.py') for p in files)
print(json.dumps(dict(status='FAILED_BEFORE_MODEL_BUILD_OR_OPTIMIZER',state=state,files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files})))
