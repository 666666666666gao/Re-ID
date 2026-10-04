from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');campaign=root/'logs/deployment_metric_role_v1_20261005_837';parent=root/'logs/deployment_metric_role_launch_20261005_837'
state=json.loads((campaign/'campaign.json').read_text());terminal=json.loads((parent/'EXIT.json').read_text())
assert state['status']=='FAILED' and 'active_command' not in state and terminal['exit_code']==1
failed=next(j for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==('MSVR310','native','m0'))
assert failed['status']=='FAILED' and failed['exit_code']==1
names=[campaign/'campaign.json',campaign/'manifest.json',campaign/'MSVR310_native_m0.log',campaign/'prepare_MSVR310_native.log',
       campaign/'initialization/MSVR310_native.json',parent/'EXIT.json',parent/'console.log',parent/'supervisor.py',parent/'LAUNCH.json']
m0=root/'trained-model/deployment_metric_role_v1_20261005_837_m0_native_MSVR310'
names.extend(p for p in m0.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.txt'))
files={str(p.relative_to(root)):dict(text=p.read_bytes().decode('utf-8'),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in names}
print(json.dumps(dict(status='ORIGINAL_QUEUE_FAILED_NATIVE_M0_INTAKE_ONLY',at=datetime.now().astimezone().isoformat(),
 campaign=state,parent_exit=terminal,failed=failed,failed_m0_files=sorted(p.name for p in m0.iterdir()),files=files,
 disk_free_bytes=shutil.disk_usage(root).free,boundary='Original terminal failure read only. No new model,retry,threshold change or power/temperature query. Completed3formal preserved;pending3 not claimed.')))
