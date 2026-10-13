"""First actual NN after recorded storage-only admission failure; do not rerun original wrapper."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

w=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (w/'CONTINUATION_EXIT.json').exists()
assert json.loads((w/'LAUNCH_EXIT.json').read_text())['exit_code']==1
assert (w/'LAUNCH.stderr').read_text().endswith('AssertionError: 2660458496\n')
archive=json.loads((w/'DISTANCE_ARCHIVE.json').read_text())
assert archive['status']=='REMOTE_DUPLICATE_RETIRED_LOCAL_CANONICAL' and archive['weights_deleted']==0
assert sha(Path(archive['local_archive']))==archive['sha256']
audit=json.loads((w/'SOURCE_AUDIT.json').read_text())
assert audit['source_readiness']=='PASS' and audit['source_blockers']==[]
assert all(sha(Path(p))==h.removeprefix('sha256:') for p,h in audit['audited_input_hashes'].items())
continuation_audit=json.loads((w/'CONTINUATION_SOURCE_AUDIT.json').read_text())
assert continuation_audit['source_readiness']=='PASS' and not continuation_audit['source_blockers']
assert all(sha(Path(p))==h.removeprefix('sha256:') for p,h in continuation_audit['audited_input_hashes'].items())
request=json.loads((w/'LAUNCH_REQUEST.json').read_text())
assert request['storage_required_bytes']==2684354560
tree=ast.parse((w/'run_diagnosis.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
body=node.value.right.value
assert body.startswith('\nfrom datetime import datetime\nimport hashlib,json,os,shutil,subprocess,time\n')
prefix=r'''
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion')
assert not (root/'logs/region_readout_use933').exists()
assert not (root/'tools/diagnose_region_readout_use.py').exists()
assert not (root/'refine-logs/region_readout_use_20261013_933/EXPERIMENT_PLAN.md').exists()
'''
code='request='+repr(request)+'\n'+prefix+body
(w/'CONTINUATION_REQUEST.json').write_text(json.dumps(dict(original_launch_exit=1,
    original_native_chunk='912b15',original_nn_calls=0,original_failure='storage only, before GPU check or install',
    original_request_sha256=sha(w/'LAUNCH_REQUEST.json'),reviewed_body_sha256=hashlib.sha256(body.encode()).hexdigest(),
    archive_sha256=sha(w/'DISTANCE_ARCHIVE.json'),new_output_namespace='logs/region_readout_use933',
    source_audit_sha256=sha(w/'SOURCE_AUDIT.json'),continuation_source_audit_sha256=sha(w/'CONTINUATION_SOURCE_AUDIT.json'),
    request_threshold_unchanged=True,nn_retry=False),indent=2)+'\n')
opts=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*opts,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],
    input=code,text=True,capture_output=True)
(w/'CONTINUATION.stdout').write_text(run.stdout,encoding='utf-8')
(w/'CONTINUATION.stderr').write_text(run.stderr,encoding='utf-8')
(w/'CONTINUATION_EXIT.json').write_text(json.dumps(dict(exit_code=run.returncode,
    at=datetime.now().astimezone().isoformat()),indent=2)+'\n')
assert run.returncode==0,run.stderr
result=json.loads(run.stdout)
(w/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['exit']['status'],'exits':result['exit']['exits'],
    'metrics':{e:{n:s['metrics'] for n,s in d['scores'].items()} for e,d in result['diagnoses'].items()}}))
