"""Retire only the six exact successful, closed prompt M0 reload probes."""
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
raw = Path('C:/Users/gb/.codex_tmp/prompt_role_state_20260930')
base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet = base / 'closed_prompt_m0_retirement834'
assert not packet.exists()
summary_path = repo / 'logs/prompt_role_state_20260930/accepted_matrix.json'
summary = json.loads(summary_path.read_bytes())
assert summary['verified_complete'] == summary['expected_endpoints'] == len(summary['rows']) == 6
assert {(r['dataset'], r['variant']) for r in summary['rows']} == {
    (d, v) for d in ('RGBNT201', 'RGBNT100', 'MSVR310') for v in ('reset_roles', 'carry_roles')}
targets = []
expected_artifacts = {}
def bind(remote_path, local_path):
    assert local_path.is_file()
    expected_artifacts[str(remote_path)] = hashlib.sha256(local_path.read_bytes()).hexdigest()
root = PurePosixPath('/data/gaob/Re-ID/Trifusion')
for row in summary['rows']:
    assert row['status'] == 'VERIFIED_COMPLETE'
    full = PurePosixPath(row['run_dir'])
    m0 = full.with_name(full.name.removesuffix('_full') + '_m0')
    receipt_path = raw / 'trained-model' / m0.name / 'training.json'
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt['status'] == 'M0_PASS' and receipt['m0']['reload_max_abs_difference'] == 0
    assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters']
    targets.append(dict(path=str(m0 / 'm0_reload_probe.pth'), sha256=receipt['m0']['reload_probe_sha256'],
                        receipt=str(m0 / 'training.json'), full_dir=str(full), campaign_dir=row['campaign_dir'],
                        checkpoint_sha256=row['checkpoint_sha256'], distance_sha256=row['distance_sha256'],
                        receipt_sha256=row['receipt_sha256']))
    bind(m0 / 'training.json', receipt_path)
    for name in ('training.json', 'official_metrics.json'):
        bind(full / name, raw / 'trained-model' / full.name / name)
    child = PurePosixPath(row['campaign_dir']) / 'campaign.json'
    bind(child, repo / child.relative_to(root).as_posix())
for name in ('campaign.json', 'accepted_matrix.json', 'paired_RGBNT201.json', 'paired_RGBNT100.json', 'paired_MSVR310.json'):
    p = root / 'logs/prompt_role_state_20260930' / name
    bind(p, repo / p.relative_to(root).as_posix())
p = root / 'results/PROMPT_ROLE_STATE_FULL_2026-09-30.md'
bind(p, repo / p.relative_to(root).as_posix())

code = "\nfrom datetime import datetime\nfrom pathlib import Path\nimport hashlib,json,shutil\nroot=Path('/data/gaob/Re-ID/Trifusion')\nsha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()\nexpected=EXPECTED_ARTIFACTS\ntargets=TARGETS\nassert len(targets)==6\nreport=root/'logs/prompt_role_state_20260930/accepted_matrix.json'\nassert sha(report)==EXPECTED_REPORT_SHA\nassert all(sha(Path(name))==digest for name,digest in expected.items())\nold=root/'logs/prompt_role_state_20260930'\nstate=json.loads((old/'campaign.json').read_text())\nmatrix=json.loads(report.read_text())\nassert state['status']=='COMPLETE' and len(state['jobs'])==6\nassert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])\nassert matrix['verified_complete']==matrix['expected_endpoints']==6\nassert len(matrix['rows'])==6 and all(r['status']=='VERIFIED_COMPLETE' for r in matrix['rows'])\nfor target in targets:\n child=json.loads((Path(target['campaign_dir'])/'campaign.json').read_text())\n assert child['status']=='COMPLETE' and len(child['jobs'])==3\n assert [j['mode'] for j in child['jobs']]==['m0','train','evaluate']\n assert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in child['jobs'])\n\ncurrent=root/'logs/global_task_role_v1_20261004_824'\nmanifest=json.loads((current/'manifest.json').read_text())\nassert len(manifest['source_sha256'])==330\nassert all(sha(root/name)==digest for name,digest in manifest['source_sha256'].items())\ncontrol=root/'refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json'\nhistorical=root/'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json'\nassert sha(control)==manifest['control_seal_sha256']\nassert sha(historical)==manifest['historical_control_seal_sha256']\nprotected=set(manifest['initialization_sha256'])\nfor p in (control,historical):\n protected.update(json.loads(p.read_text())['artifact_sha256'])\nassert not protected.intersection(t['path'] for t in targets)\ncurrent_state=json.loads((current/'campaign.json').read_text())\nprobes={}\nfor j in current_state['jobs']:\n if j['phase']=='m0':\n  assert j['status']=='COMPLETE' and j['exit_code']==0\n  p=root/'trained-model'/('global_task_role_v1_20261004_824_m0_'+j['variant']+'_'+j['dataset'])/'m0_reload_probe.pth'\n  digest=j['result']['m0']['reload_probe_sha256']\n  assert sha(p)==digest\n  probes[str(p)]=digest\nassert len(probes)==6\nprocess=Path('/proc/3215669/stat').read_text()\nassert int(process[process.rfind(')')+2:].split()[19])==38149139\ncommand=Path('/proc/3215669/cmdline').read_bytes().replace(b'\\0',b' ')\nassert b'run_global_task_role.py' in command and b'RGBNT100' in command and b'native' in command\n\nretiring=[]\nbest={}\nfor t in targets:\n p=Path(t['path'])\n assert p.resolve().is_relative_to((root/'trained-model').resolve()) and p.resolve()==p\n assert p.name=='m0_reload_probe.pth' and p.parent.name.startswith('prompt_role_state_20260930_prompt_') and p.parent.name.endswith('_m0')\n assert sha(p)==t['sha256']\n receipt=json.loads(Path(t['receipt']).read_text())\n assert receipt['status']=='M0_PASS' and receipt['m0']['reload_probe_sha256']==t['sha256']\n full=Path(t['full_dir'])\n assert [p.name for p in full.glob('*.pth')]==['best_map.pth']\n training=json.loads((full/'training.json').read_text())\n assert [r['epoch'] for r in training['history']]==list(range(1,51))\n official=json.loads((full/'official_metrics.json').read_text())\n assert official['status']=='COMPLETE' and official['training_epochs']==50\n for name,key in (('best_map.pth','checkpoint_sha256'),('official_distances.pt','distance_sha256'),('official_metrics.json','receipt_sha256')):\n  actual=sha(full/name)\n  assert actual==t[key]\n  best[str(full/name)]=actual\n retiring.append(dict(t,bytes=p.stat().st_size))\njournal=root/'logs/prompt_m0_retirement834_20261005'\nassert not journal.exists()\njournal.mkdir()\nfree_before=shutil.disk_usage(root).free\nplan=dict(status='CLOSED_SIX_ENDPOINT_M0_PROBES_VERIFIED_BEFORE_RETIREMENT',\n at=datetime.now().astimezone().isoformat(),targets=retiring,formal_artifacts=best,\n current_required_probes=probes,original_report_sha256=sha(report),\n protected_control_seal_sha256={str(p):sha(p) for p in (control,historical)},\n source_count=330,active_training_pid=3215669,active_training_start_ticks=38149139,\n boundary='Only six exact successful M0 reload probes; full50, firststrict and complete-gallery verification completed 2026-09-30. Formal best/receipts and current six required probes retained. No model forward, original verifier/report replay or power/temperature action.')\n(journal/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\\n')\nfor t in retiring:\n p=Path(t['path'])\n assert sha(p)==t['sha256']\n p.unlink()\n assert not p.exists()\n with (journal/'RETIREMENT.jsonl').open('a') as stream:\n  stream.write(json.dumps(dict(path=t['path'],bytes=t['bytes'],sha256=t['sha256'],retired_at=datetime.now().astimezone().isoformat()))+'\\n')\nassert all(sha(Path(name))==digest for name,digest in best.items())\nassert all(sha(Path(name))==digest for name,digest in probes.items())\nassert all(sha(Path(name))==digest for name,digest in expected.items())\nassert all(sha(root/name)==digest for name,digest in manifest['source_sha256'].items())\nassert sha(control)==manifest['control_seal_sha256'] and sha(historical)==manifest['historical_control_seal_sha256']\nprocess_after=Path('/proc/3215669/stat').read_text()\nassert int(process_after[process_after.rfind(')')+2:].split()[19])==38149139\nresult=dict(status='SIX_CLOSED_PROMPT_M0_PROBES_RETIRED',completed_at=datetime.now().astimezone().isoformat(),\n retired_files=6,retired_bytes=sum(t['bytes'] for t in retiring),disk_free_before=free_before,\n disk_free_after=shutil.disk_usage(root).free,formal_best_and_receipts_unchanged=True,\n current_six_probes_unchanged=True,current_scientific330_sources_unchanged=True,\n current_native_pid_still_same=True,\n boundary='Old probe binaries no longer available for direct M0-replay. Original successful receipts, formal best and full reports remain. Disk free delta includes concurrent training writes; retired_bytes is exact file-byte sum.')\n(journal/'RESULT.json').write_text(json.dumps(result,indent=2)+'\\n')\nprint(json.dumps(dict(result=result,plan=plan,journal={p.name:p.read_text() for p in journal.iterdir()})))\n"
for marker, value in (('EXPECTED_ARTIFACTS', repr(expected_artifacts)), ('TARGETS', repr(targets)),
                      ('EXPECTED_REPORT_SHA', repr(hashlib.sha256(summary_path.read_bytes()).hexdigest()))):
    assert marker in code
    code = code.replace(marker, value)
compile(code, 'remote_retire_closed_prompt_m0.py', 'exec')
packet.mkdir()
(packet / 'remote_retire_closed_prompt_m0.py').write_text(code, encoding='utf-8')
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
