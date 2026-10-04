
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='dba87a3864e46ec20c2957098f6fb9f653054cb6'
paths=['refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json', 'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json']
digests=['e997d2341ca86e875a6518162965182895a7778fbd84a99c7a3ffa70f82776e5', 'faec05a5eea483570b136a1c087e787c336f5fee2cf9d416b05c879d0bcee724']
for name,digest in zip(paths,digests):
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
current=json.loads((root/paths[0]).read_text())
old=json.loads((root/paths[1]).read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in current['source_sha256'].items())
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in current['artifact_sha256'].items())
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in old['artifact_sha256'].items())
parent=json.loads((root/'logs/role_input_detach_launch_20261004_813/EXIT.json').read_text())
state=json.loads((root/'logs/role_input_detach_v1_20261004_813/campaign.json').read_text())
assert parent['exit_code']==0 and state['status']=='COMPLETE' and state['report_exit_code']==0
print(json.dumps(dict(status='EXISTING_SOURCE324_AND_FORMAL_CONTROLS_VERIFIED',at=datetime.now().astimezone().isoformat(),
 source_files=324,current_artifacts=124,historical_artifacts=len(old['artifact_sha256']),
 disk_free_bytes=shutil.disk_usage(root).free,retired_m0_accessed=False)))
