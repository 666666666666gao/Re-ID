"""Seal the sole next training intervention against existing formal controls."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private = Path('C:/Users/gb/.codex_tmp')
packet = private / 'independent_evidence_draft/global_task_role_source_seal'
assert not packet.exists()
proof = json.loads((private / 'foundation_recipe_v1_20261002/four_copy823_2025_pending.json').read_bytes())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == proof['head']
sealpath = 'refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json'
historical = 'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json'
sealed = json.loads((repo / sealpath).read_bytes())
assert len(sealed['source_sha256']) == 324 and len(sealed['artifact_sha256']) == 124
new_names = ['modeling/trifusion/global_task_role_heads.py', 'tools/run_global_task_role.py',
             'tools/queue_global_task_role.py', 'tools/report_global_task_role.py',
             'tools/check_global_task_role_heads.py', 'refine-logs/global_task_role_v1/EXPERIMENT_PLAN.md']
new_sources = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in new_names}
for name in new_names:
    if name.endswith('.py'):
        ast.parse((repo / name).read_text(encoding='utf-8'))
cpu = private / 'independent_evidence_draft'
assert json.loads((cpu / 'global_task_role_configuration/EXIT.json').read_bytes())['entry_sha256'] == new_sources['tools/run_global_task_role.py']
for folder in ('global_task_role_configuration', 'global_task_role_cpu_witness'):
    assert json.loads((cpu / folder / 'EXIT.json').read_bytes())['exit_code'] == 0
model_witness = json.loads((cpu / 'global_task_role_cpu_witness/EXIT.json').read_bytes())['staged_source_sha256']
assert model_witness['trifusion/global_task_role_heads.py'] == new_sources[new_names[0]]
assert model_witness['check_global_task_role_heads.py'] == new_sources[new_names[4]]
code = f'''
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=={proof['head']!r}
paths={[sealpath,historical]!r}
digests={[hashlib.sha256((repo / n).read_bytes()).hexdigest() for n in (sealpath,historical)]!r}
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
'''
compile(code, 'remote_source_seal.py', 'exec')
packet.mkdir()
(packet / 'remote_source_seal.py').write_text(code, encoding='utf-8')
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(180)
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_text(json.dumps(dict(exit_code=exit_code)) + '\n', encoding='utf-8')
client.close()
assert exit_code == 0, error.decode()
sources = {**sealed['source_sha256'], **new_sources}
assert len(sources) == 330
scope = dict(schema='trifusion-global-task-role-source-scope-v1',registered_at=datetime.now().astimezone().isoformat(),
 source_sha256=sources,control_seal_sha256=hashlib.sha256((repo / sealpath).read_bytes()).hexdigest(),
 historical_control_seal_sha256=hashlib.sha256((repo / historical).read_bytes()).hexdigest(),
 cpu_component_and_configuration_checked=True,
 boundary='Original324 sources unchanged remotely; six new files pinned locally. CPU ownership/configuration evidence only, no production prepare/M0/full50 yet. Retired M0 binaries not accessed.')
target = repo / 'refine-logs/global_task_role_v1/SOURCE_SCOPE.json'
assert not target.exists()
target.write_text(json.dumps(scope, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status='SOURCE330_SEALED',new_sources=new_sources,scope_sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
