from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/semantic_capacity_first_eval_completion_20261005_857'
launch=root/'logs/semantic_capacity_first_eval_launch_20261005_857'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((campaign/'PLAN.json').read_text())
assert all(sha(Path(n))==d for n,d in plan['original_failure_sha256'].items())
exit_path=launch/'EXIT.json'
exit_record=json.loads(exit_path.read_text()) if exit_path.exists() else None
state=json.loads((campaign/'campaign.json').read_text())
if exit_record is None:
 assert Path('/proc/1171133/stat').read_text().split()[21]=='43648534'
logs={p.name:p.read_text()[-5000:] for p in (launch/'console.log',campaign/'RGBNT100_first_evaluate.log',campaign/'report.log') if p.is_file()}
value=dict(at=datetime.now().astimezone().isoformat(),campaign=state,exit=exit_record,
 accepted=[p.name for p in (campaign/'acceptance').glob('*.json')],
 original_parent_exit_code=1,new_training_invocations=state['new_training_invocations'],
 original_failure_sha256=plan['original_failure_sha256'],logs=logs,disk_free_bytes=shutil.disk_usage(root).free)
print(json.dumps(value))
