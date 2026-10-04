"""Wait locally until the original native endpoint's estimated finishing window."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet = base / 'global_task_role_rgb100_native_milestone831'
assert not packet.exists()
summary = json.loads((base / 'global_task_role_rgb100_semantic_full829/SUMMARY.json').read_bytes())
job = next(j for j in summary['campaign_state']['jobs'] if (j['dataset'], j['variant'], j['phase']) == ('RGBNT100', 'native', 'full'))
step = next(s for s in job['steps'] if s['mode'] == 'train')
assert step['status'] == 'RUNNING' and step['gpus'] == [0, 1]
target = datetime.fromisoformat('2026-10-05T01:00:00+08:00')
started = datetime.now().astimezone()
delay = (target - started).total_seconds()
assert delay > 0
packet.mkdir()
(packet / 'START.json').write_text(json.dumps(dict(
    started_at=started.isoformat(), target_at=target.isoformat(), delay_seconds=delay,
    original_training_started_at=step['started_at'], original_training_pid=step['pid'],
    expected_duration_source='Previous complete RGBNT100 native full training plus epoch evaluations: 7252.288336 seconds; current semantic: 7126.702006 seconds.',
    boundary='Local wait only; one original observer at the estimated finishing window, no remote poll before then or model restart.'
), indent=2) + '\n', encoding='utf-8')
time.sleep(delay)
with (packet / 'stdout.json').open('x', encoding='utf-8') as out, (packet / 'stderr.txt').open('x', encoding='utf-8') as err:
    result = subprocess.run(['uv', 'run', '--with', 'paramiko', 'python', '-X', 'utf8',
                             '.codex_tmp/observe_global_task_role.py', '0'],
                            cwd='C:/Users/gb', stdout=out, stderr=err)
(packet / 'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,
    completed_at=datetime.now().astimezone().isoformat())) + '\n', encoding='utf-8')
raise SystemExit(result.returncode)
