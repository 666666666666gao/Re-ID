from datetime import datetime
from pathlib import Path
import ast,hashlib,json,subprocess
import paramiko

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');private=Path('C:/Users/gb/.codex_tmp')
packet=private/'independent_evidence_draft/deployment_metric_source_seal837';assert not packet.exists()
proof=json.loads((private/'foundation_recipe_v1_20261002/four_copy836_2025_pending.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==proof['head']
sealname='refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
old=json.loads((repo/sealname).read_bytes());assert len(old['source_sha256'])==332 and len(old['artifact_sha256'])==187
names=['modeling/trifusion/deployment_metric_role.py','tools/run_deployment_metric_role.py',
       'tools/check_deployment_metric_role.py','tools/queue_deployment_metric_role.py','tools/report_deployment_metric_role.py',
       'refine-logs/deployment_metric_role_v1/EXPERIMENT_PLAN.md','refine-logs/deployment_metric_role_v1/EXPERIMENT_CODE_REVIEW.md']
new={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in names}
checks=private/'independent_evidence_draft/deployment_metric_checks837'
for filename in ('CONFIG_EXIT.json','WITNESS_EXIT.json'):
    receipt=json.loads((checks/filename).read_bytes());assert receipt['exit_code']==0
    assert all(new[n]==digest for n,digest in receipt['sources'].items())
for name in names:
    if name.endswith('.py'):ast.parse((repo/name).read_text(encoding='utf-8'))
code=f'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=={proof['head']!r}
assert hashlib.sha256((root/{sealname!r}).read_bytes()).hexdigest()=={hashlib.sha256((repo/sealname).read_bytes()).hexdigest()!r}
bound=json.loads((root/{sealname!r}).read_text())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in bound['source_sha256'].items())
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==d for n,d in bound['artifact_sha256'].items())
for name in ['logs/global_task_role_launch_20261004_824/EXIT.json','logs/global_task_role_fixed_best_launch_20261004_v1/EXIT.json']:
 assert json.loads((root/name).read_text())['exit_code']==0
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in {dict((n,d) for n,d in new.items() if not n.endswith('EXPERIMENT_CODE_REVIEW.md'))!r}.items())
assert shutil.disk_usage(root).free>=7*384*1024**2+2*1024**3
print(json.dumps(dict(status='ORIGINAL332_SOURCE_AND187_CONTROL_ARTIFACTS_VERIFIED',at=datetime.now().astimezone().isoformat(),source_files=332,control_artifacts=187,retired_probe_accessed=False,disk_free_bytes=shutil.disk_usage(root).free)))
'''
packet.mkdir();(packet/'remote_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n')
client.close();assert rc==0,error.decode()
scope=dict(schema='trifusion-deployment-metric-role-source-scope-v1',registered_at=datetime.now().astimezone().isoformat(),
           source_sha256={**old['source_sha256'],**new},control_seal_sha256=hashlib.sha256((repo/sealname).read_bytes()).hexdigest(),
           cpu_checks={n:hashlib.sha256((checks/n).read_bytes()).hexdigest() for n in ('CONFIG_EXIT.json','WITNESS_EXIT.json')},
           boundary='Original332source and187 controls unchanged,7newsource/plan/review files pinned. CPU checks only; no actual prepare/M0/full50 yet. No retired-probe access, power/temperature actions or2025 access.')
assert len(scope['source_sha256'])==339
target=repo/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json';assert not target.exists()
target.write_text(json.dumps(scope,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='SOURCE339_SEALED',scope_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),new_files=new)))
