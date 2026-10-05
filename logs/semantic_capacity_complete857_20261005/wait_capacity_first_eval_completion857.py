"""Observe the sole first-evaluation completion near its expected terminal time."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import paramiko

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet = base / 'capacity_first_eval_completion_milestone857'
launch = json.loads((base / 'capacity_first_eval_completion_launch857/stdout.json').read_bytes())
target = datetime.fromisoformat(launch['at']) + timedelta(seconds=180)
assert not packet.exists()
packet.mkdir()
start = dict(status='WAITING_SOLE_FIRST_EVAL_COMPLETION', at=datetime.now().astimezone().isoformat(),
             first_observation_at=target.isoformat(), poll_seconds=240, pid=launch['pid'],
             expected_start_ticks=launch['start_ticks'], new_training_invocations=0)
(packet / 'START.json').write_bytes((json.dumps(start, indent=2) + '\n').encode())
print(json.dumps(start), flush=True)
code = rf'''from pathlib import Path
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
 assert Path('/proc/{launch['pid']}/stat').read_text().split()[21]=={launch['start_ticks']!r}
logs={{p.name:p.read_text()[-5000:] for p in (launch/'console.log',campaign/'RGBNT100_first_evaluate.log',campaign/'report.log') if p.is_file()}}
value=dict(at=datetime.now().astimezone().isoformat(),campaign=state,exit=exit_record,
 accepted=[p.name for p in (campaign/'acceptance').glob('*.json')],
 original_parent_exit_code=1,new_training_invocations=state['new_training_invocations'],
 original_failure_sha256=plan['original_failure_sha256'],logs=logs,disk_free_bytes=shutil.disk_usage(root).free)
print(json.dumps(value))
'''
compile(code, 'capacity_first_eval_completion_observer857.py', 'exec')
(packet / 'REMOTE_SOURCE.py').write_bytes(code.encode())
while datetime.now().astimezone() < target:
    time.sleep(min(240, (target - datetime.now().astimezone()).total_seconds()))
index = 0
while True:
    client = paramiko.SSHClient()
    client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
    client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
    stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
    stdin.write(code)
    stdin.channel.shutdown_write()
    stdout.channel.settimeout(60)
    data, error = stdout.read(), stderr.read()
    status = stdout.channel.recv_exit_status()
    client.close()
    index += 1
    (packet / f'observation{index:02}.json').write_bytes(data)
    (packet / f'stderr{index:02}.txt').write_bytes(error)
    assert status == 0, error.decode()
    value = json.loads(data)
    state = value['campaign']
    brief = dict(at=value['at'],status=state['status'],exit=value['exit'],
                 active=state.get('active_command',{}).get('mode'),accepted=value['accepted'],
                 report_invocations=state['report_invocations'],report_exit_code=state.get('report_exit_code'),
                 original_parent_exit_code=1,new_training_invocations=0,disk_free_bytes=value['disk_free_bytes'])
    print(json.dumps(brief), flush=True)
    if value['exit'] is not None:
        (packet / 'FINAL.json').write_bytes(data)
        (packet / 'EXIT.json').write_bytes((json.dumps(dict(exit_code=0, observed=brief,
            observer_completed_at=datetime.now().astimezone().isoformat()))+'\n').encode())
        break
    time.sleep(240)
