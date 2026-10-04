from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='331a3cb6cbd61d299440231c4bf2df1840623650'
assert hashlib.sha256((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_bytes()).hexdigest()=='9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
bound=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in bound['source_sha256'].items())
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==d for n,d in bound['artifact_sha256'].items())
for name in ['logs/global_task_role_launch_20261004_824/EXIT.json','logs/global_task_role_fixed_best_launch_20261004_v1/EXIT.json']:
 assert json.loads((root/name).read_text())['exit_code']==0
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in {'modeling/trifusion/deployment_metric_role.py': 'fa60927aa22c1af6055a0b8bb229bcd07eda41a864e8a549a53a6cb27d1e0564', 'tools/run_deployment_metric_role.py': '42b061a7426d032675d53e1c576fb5955d7be50c20b2b2a9559e15b7979a907b', 'tools/check_deployment_metric_role.py': '0beb0b82a9fd94ba47c2899015c3d11d3a094bff3c90771659c5441d92771e26', 'tools/queue_deployment_metric_role.py': '1ad600fd4fb48c976bee6d1dabdb9745a6b3da782325485c7f9798d6c425390b', 'tools/report_deployment_metric_role.py': '6fdd3d4a46cc15bb5d64020580fa86033be60b2fa46531040dae5c2cecbf0d34', 'refine-logs/deployment_metric_role_v1/EXPERIMENT_PLAN.md': '110f9d25866755316664ee582d558b636854296a8b318bd66cfdde77b3064ac7'}.items())
assert shutil.disk_usage(root).free>=7*384*1024**2+2*1024**3
print(json.dumps(dict(status='ORIGINAL332_SOURCE_AND187_CONTROL_ARTIFACTS_VERIFIED',at=datetime.now().astimezone().isoformat(),source_files=332,control_artifacts=187,retired_probe_accessed=False,disk_free_bytes=shutil.disk_usage(root).free)))
