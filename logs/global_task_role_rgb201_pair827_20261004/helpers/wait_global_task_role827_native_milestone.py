"""Observe the original native job once near its estimated completion."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/global_task_role827_native_milestone')
assert not packet.exists()
packet.mkdir()
deadline = datetime.fromisoformat('2026-10-04T20:04:00+08:00')
start = dict(status='WAITING_ESTIMATED_NATIVE_ENDPOINT_MILESTONE',
             started_at=datetime.now().astimezone().isoformat(), deadline=deadline.isoformat(),
             pid=__import__('os').getpid(), campaign='logs/global_task_role_v1_20261004_824',
             last_verified_live_training_pid=2761377, last_verified_start_ticks=36846190,
             boundary='Local scheduled observer only. Native fresh50 started19:25:21; estimated full completion20:05–20:09. No remote reads before deadline, no restart or power/temperature action.')
(packet / 'START.json').write_text(json.dumps(start, indent=2) + '\n', encoding='utf-8')
print(json.dumps(start), flush=True)
while datetime.now().astimezone() < deadline:
    time.sleep(min(60, (deadline - datetime.now().astimezone()).total_seconds()))
with (packet / 'stdout.json').open('x', encoding='utf-8') as out, (packet / 'stderr.txt').open('x', encoding='utf-8') as err:
    result = subprocess.run([sys.executable, '-X', 'utf8', 'C:/Users/gb/.codex_tmp/observe_global_task_role.py', '0'],
                            cwd='C:/Users/gb', stdout=out, stderr=err)
(packet / 'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,
    completed_at=datetime.now().astimezone().isoformat())) + '\n', encoding='utf-8')
print(json.dumps(dict(status='NATIVE_MILESTONE_OBSERVER_RETURNED', exit_code=result.returncode, packet=str(packet))), flush=True)
raise SystemExit(result.returncode)
