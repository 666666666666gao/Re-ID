"""Register the prepared fixed-best diagnostic after the original intake closed."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private = Path('C:/Users/gb/.codex_tmp')
base = private / 'independent_evidence_draft'
packet = base / 'global_task_role_fixed_best_registration835'
destination = repo / 'refine-logs/global_task_role_fixed_best_diagnosis_v1'
candidate = private / 'global_task_role_fixed_best_candidate_v2'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
previous = json.loads((private/'foundation_recipe_v1_20261002/four_copy834_2025_pending.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==previous['head']
assert not packet.exists() and not destination.exists()
intake = base / 'global_task_role_complete_report'
assert json.loads((intake/'EXIT.json').read_bytes())['exit_code']==0
record = json.loads((intake/'stdout.json').read_bytes())
assert record['status']=='ORIGINAL_ONCE_ONLY_FINAL_CPU_REPORT_AND_ARTIFACTS_VERIFIED'
assert record['formal_completed']==6 and record['formal_epochs']==300 and record['formal_steps']==12968
manifest = json.loads((intake/'texts/global_task_role_v1_20261004_824_manifest.json').read_bytes())
source = dict(manifest['source_sha256'])
assert len(source)==330
destination.mkdir()
wrapper = 'tools/diagnose_global_task_role_best.py'
plan = 'refine-logs/global_task_role_fixed_best_diagnosis_v1/EXPERIMENT_PLAN.md'
assert not (repo/wrapper).exists()
shutil.copyfile(candidate/'tools/diagnose_global_task_role_best.py',repo/wrapper)
shutil.copyfile(candidate/'EXPERIMENT_PLAN.md',repo/plan)
assert sha(repo/wrapper)=='e9cff6ae84b3e846cdd40d8b9f89581dccf0970e1cc6181c5eae7b449420a64d'
assert sha(repo/plan)=='f8a2183dc4b3ae7adf512286766c6d98a941de4b3ed8fcb1a5b7f659bd7864ed'
source.update({name:sha(repo/name) for name in (wrapper,plan)})
scope = dict(schema='trifusion-global-task-role-fixed-best-source-v1',registered_at=datetime.now().astimezone().isoformat(),
             source_sha256=source,source_count=332,original_source_count=330,
             boundary='All six original full50/first-strict and original once CPU report received. Prepared v2 diagnostic only; no new training or model changes.')
scope_name='refine-logs/global_task_role_fixed_best_diagnosis_v1/SOURCE_SCOPE.json'
(repo/scope_name).write_text(json.dumps(scope,indent=2)+'\n',encoding='utf-8')
packet.mkdir()
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
root='/data/gaob/Re-ID/Trifusion'
code=f"from pathlib import Path;import hashlib,json;root=Path({root!r});expected={manifest['source_sha256']!r};assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in expected.items());assert not (root/{wrapper!r}).exists();assert not (root/{plan!r}).parent.exists();(root/{plan!r}).parent.mkdir();print(json.dumps(dict(status='ORIGINAL330_VERIFIED_BEFORE_DIAGNOSTIC_REGISTRATION')))"
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write()
data,error=stdout.read(),stderr.read();exit_code=stdout.channel.recv_exit_status()
(packet/'precheck_stdout.json').write_bytes(data);(packet/'precheck_stderr.txt').write_bytes(error)
assert exit_code==0,error.decode()
sftp=client.open_sftp()
for name in (wrapper,plan,scope_name):
    sftp.put(str(repo/name),root+'/'+name)
sftp.close()
code=f"from pathlib import Path;from datetime import datetime;import hashlib,json;root=Path({root!r});expected={source!r};actual={{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in expected}};assert actual==expected;assert hashlib.sha256((root/{scope_name!r}).read_bytes()).hexdigest()=={sha(repo/scope_name)!r};print(json.dumps(dict(status='FIXED_BEST332_SOURCE_REGISTERED_UNEXECUTED',at=datetime.now().astimezone().isoformat(),source_count=len(actual),source_scope_sha256={sha(repo/scope_name)!r},wrapper_sha256={source[wrapper]!r},plan_sha256={source[plan]!r})))"
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write()
data,error=stdout.read(),stderr.read();exit_code=stdout.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=exit_code))+'\n',encoding='utf-8')
client.close()
assert exit_code==0,error.decode()
print(data.decode())
