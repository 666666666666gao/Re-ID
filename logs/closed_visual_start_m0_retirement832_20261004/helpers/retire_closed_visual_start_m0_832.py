"""Retire only the six successful, closed visual-start M0 reload probes."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
raw = repo / 'logs/visual_start_complete_700_20261001/raw'
base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet = base / 'closed_visual_start_m0_retirement832'
assert not packet.exists()
summary_path = raw / 'results/visual_start_roles_complete_20261001/SUMMARY.json'
summary = json.loads(summary_path.read_bytes())
assert summary['accepted'] == 6 and len(summary['rows']) == 6
assert {(r['dataset'], r['variant']) for r in summary['rows']} == {
    (d, v) for d in ('RGBNT201', 'RGBNT100', 'MSVR310') for v in ('public_visual', 'reid_visual')}
targets = []
for row in summary['rows']:
    assert row['status'] == 'VERIFIED_COMPLETE'
    full = Path(row['run_dir'])
    m0 = full.with_name(full.name.removesuffix('_full') + '_m0')
    receipt = json.loads((raw / 'trained-model' / m0.name / 'training.json').read_bytes())
    assert receipt['status'] == 'M0_PASS' and receipt['m0']['reload_max_abs_difference'] == 0
    assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters']
    targets.append(dict(path=str(m0 / 'm0_reload_probe.pth'), sha256=receipt['m0']['reload_probe_sha256'],
                        receipt=str(m0 / 'training.json'), full_dir=str(full),
                        checkpoint_sha256=row['checkpoint_sha256'], distance_sha256=row['distance_sha256'],
                        receipt_sha256=row['receipt_sha256']))
expected_artifacts = summary['source_artifacts_sha256']
assert all(hashlib.sha256((raw / Path(p).relative_to('/data/gaob/Re-ID/Trifusion')).read_bytes()).hexdigest() == digest
           for p, digest in expected_artifacts.items())

code = '''
from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected=EXPECTED_ARTIFACTS
targets=TARGETS
assert len(targets)==6
report=root/'results/visual_start_roles_complete_20261001/SUMMARY.json'
assert sha(report)==EXPECTED_REPORT_SHA
assert all(sha(Path(name))==digest for name,digest in expected.items())
old=root/'logs/visual_start_roles_20261001_v1'
state=json.loads((old/'campaign.json').read_text())
analysis=json.loads((old/'analysis_waiter_status.json').read_text())
assert state['status']=='COMPLETE' and len(state['jobs'])==6
assert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
assert analysis['status']=='CPU_REPORT_COMPLETE' and analysis['exit_code']==0
assert analysis['complete_parent_jobs']==analysis['expected_jobs']==6

current=root/'logs/global_task_role_v1_20261004_824'
manifest=json.loads((current/'manifest.json').read_text())
assert len(manifest['source_sha256'])==330
assert all(sha(root/name)==digest for name,digest in manifest['source_sha256'].items())
control=root/'refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json'
historical=root/'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(control)==manifest['control_seal_sha256']
assert sha(historical)==manifest['historical_control_seal_sha256']
protected=set(manifest['initialization_sha256'])
for p in (control,historical):
 protected.update(json.loads(p.read_text())['artifact_sha256'])
assert not protected.intersection(t['path'] for t in targets)
current_state=json.loads((current/'campaign.json').read_text())
probes={}
for j in current_state['jobs']:
 if j['phase']=='m0':
  assert j['status']=='COMPLETE' and j['exit_code']==0
  p=root/'trained-model'/('global_task_role_v1_20261004_824_m0_'+j['variant']+'_'+j['dataset'])/'probe_checkpoint.pth'
  digest=j['result']['m0']['reload_probe_sha256']
  assert sha(p)==digest
  probes[str(p)]=digest
assert len(probes)==6
process=Path('/proc/3215669/stat').read_text()
assert int(process[process.rfind(')')+2:].split()[19])==38149139
command=Path('/proc/3215669/cmdline').read_bytes().replace(b'\\0',b' ')
assert b'run_global_task_role.py' in command and b'RGBNT100' in command and b'native' in command

retiring=[]
best={}
for t in targets:
 p=Path(t['path'])
 assert p.resolve().is_relative_to((root/'trained-model').resolve()) and p.resolve()==p
 assert p.name=='m0_reload_probe.pth' and p.parent.name.startswith('visual_start_roles_20261001_v1_visual_start_') and p.parent.name.endswith('_m0')
 assert sha(p)==t['sha256']
 receipt=json.loads(Path(t['receipt']).read_text())
 assert receipt['status']=='M0_PASS' and receipt['m0']['reload_probe_sha256']==t['sha256']
 full=Path(t['full_dir'])
 assert [p.name for p in full.glob('*.pth')]==['best_map.pth']
 training=json.loads((full/'training.json').read_text())
 assert [r['epoch'] for r in training['history']]==list(range(1,51))
 official=json.loads((full/'official_metrics.json').read_text())
 assert official['status']=='COMPLETE' and official['training_epochs']==50
 for name,key in (('best_map.pth','checkpoint_sha256'),('official_distances.pt','distance_sha256'),('official_metrics.json','receipt_sha256')):
  actual=sha(full/name)
  assert actual==t[key]
  best[str(full/name)]=actual
 retiring.append(dict(t,bytes=p.stat().st_size))
journal=root/'logs/visual_start_m0_retirement832_20261004'
assert not journal.exists()
journal.mkdir()
free_before=shutil.disk_usage(root).free
plan=dict(status='CLOSED_SIX_ENDPOINT_M0_PROBES_VERIFIED_BEFORE_RETIREMENT',
 at=datetime.now().astimezone().isoformat(),targets=retiring,formal_artifacts=best,
 current_required_probes=probes,original_report_sha256=sha(report),
 protected_control_seal_sha256={str(p):sha(p) for p in (control,historical)},
 source_count=330,active_training_pid=3215669,active_training_start_ticks=38149139,
 boundary='Only six exact successful M0 reload probes; full50, firststrict and CPU report completed 2026-10-01. Formal best/receipts and current six required probes retained. No model forward, original verifier/report replay or power/temperature action.')
(journal/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\\n')
for t in retiring:
 p=Path(t['path'])
 assert sha(p)==t['sha256']
 p.unlink()
 assert not p.exists()
 with (journal/'RETIREMENT.jsonl').open('a') as stream:
  stream.write(json.dumps(dict(path=t['path'],bytes=t['bytes'],sha256=t['sha256'],retired_at=datetime.now().astimezone().isoformat()))+'\\n')
assert all(sha(Path(name))==digest for name,digest in best.items())
assert all(sha(Path(name))==digest for name,digest in probes.items())
assert all(sha(Path(name))==digest for name,digest in expected.items())
assert all(sha(root/name)==digest for name,digest in manifest['source_sha256'].items())
assert sha(control)==manifest['control_seal_sha256'] and sha(historical)==manifest['historical_control_seal_sha256']
process_after=Path('/proc/3215669/stat').read_text()
assert int(process_after[process_after.rfind(')')+2:].split()[19])==38149139
result=dict(status='SIX_CLOSED_VISUAL_START_M0_PROBES_RETIRED',completed_at=datetime.now().astimezone().isoformat(),
 retired_files=6,retired_bytes=sum(t['bytes'] for t in retiring),disk_free_before=free_before,
 disk_free_after=shutil.disk_usage(root).free,formal_best_and_receipts_unchanged=True,
 current_six_probes_unchanged=True,current_scientific330_sources_unchanged=True,
 current_native_pid_still_same=True,
 boundary='Old probe binaries no longer available for direct M0-replay. Original successful receipts, formal best and full reports remain. Disk free delta includes concurrent training writes; retired_bytes is exact file-byte sum.')
(journal/'RESULT.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(dict(result=result,plan=plan,journal={p.name:p.read_text() for p in journal.iterdir()})))
'''
for marker, value in (('EXPECTED_ARTIFACTS', repr(expected_artifacts)), ('TARGETS', repr(targets)),
                      ('EXPECTED_REPORT_SHA', repr(hashlib.sha256(summary_path.read_bytes()).hexdigest()))):
    assert marker in code
    code = code.replace(marker, value)
compile(code, 'remote_retire_closed_visual_start_m0.py', 'exec')
packet.mkdir()
(packet / 'remote_retire_closed_visual_start_m0.py').write_text(code, encoding='utf-8')
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
(packet / 'EXIT.json').write_text(json.dumps(dict(exit_code=exit_code,at=datetime.now().astimezone().isoformat())) + '\n', encoding='utf-8')
client.close()
assert exit_code == 0, error.decode()
record = json.loads(data)
for name, content in record['journal'].items():
    (packet / name).write_text(content, encoding='utf-8')
print(json.dumps(record['result']))
