"""Read the original controller near fourth formal completion, then every240s."""
from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
import sys
import time

private = Path('C:/Users/gb/.codex_tmp/training_feature_scale_split755')
eta = json.loads((private/'LIVE_PROGRESS_ETA757.json').read_bytes())
out = private/'fourth_full_observer759'
assert not out.exists()
out.mkdir()
due = datetime.fromisoformat('2026-10-03T08:48:00+08:00')
record = {'status': 'WAITING_FOURTH_FORMAL_MILESTONE', 'pid': os.getpid(),
          'due': due.isoformat(), 'poll_seconds': 240, 'controller_pid': eta['controller_pid'],
          'boundary': 'Read-only original-controller observations; no model, score, training, restart or retry.'}

def save():
    record['updated_at'] = datetime.now().astimezone().isoformat()
    (out/'STATUS.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')

save()
while datetime.now().astimezone() < due:
    time.sleep(min(240, (due-datetime.now().astimezone()).total_seconds()))
index = 0
while True:
    index += 1
    label = f'fourth_full759_{index:04d}'
    observed = subprocess.run([sys.executable, '-X', 'utf8',
                               'C:/Users/gb/.codex_tmp/observe_metric_scale755.py', '--label', label],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out/f'{label}.stdout.txt').write_bytes(observed.stdout)
    (out/f'{label}.stderr.txt').write_bytes(observed.stderr)
    record.update(last_label=label, last_observer_exit=observed.returncode)
    if observed.returncode:
        record['status'] = 'OBSERVATION_FAILED_NO_RESTART'
        save()
        raise SystemExit(observed.returncode)
    path = private/'observations'/label/'STATUS.json'
    actual = json.loads(path.read_bytes())
    state = actual['files'].get('campaign.json', {})
    full = [row for row in state.get('jobs', []) if row['phase'] == 'full']
    complete = [row for row in full if row['status'] == 'COMPLETE' and row['exit_code'] == 0]
    record.update(last_receipt=str(path), actual_campaign_status=state.get('status'),
                  completed_formal_jobs=len(complete), controller_live=actual['controller_live'])
    if state.get('status') == 'FAILED' or not actual['controller_live']:
        record['status'] = 'CONTROLLER_TERMINAL_OBSERVED'
        save()
        break
    if len(complete) >= 4:
        record['status'] = 'FOURTH_FORMAL_COMPLETE_OBSERVED'
        save()
        break
    record['status'] = 'WAITING_FOURTH_FORMAL_COMPLETE'
    save()
    time.sleep(240)
print(json.dumps(record, indent=2))
