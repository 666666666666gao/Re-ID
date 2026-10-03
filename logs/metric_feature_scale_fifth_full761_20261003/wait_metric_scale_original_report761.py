"""Observe original F3 through full six-arm and once-report terminal closure."""
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import subprocess
import sys
import time

private = Path('C:/Users/gb/.codex_tmp/training_feature_scale_split755')
timing = json.loads(Path('C:/Users/gb/.codex_tmp/metric_feature_scale_last_run_timing761/WALL_TIMING.json').read_bytes())
assert timing['controller_pid'] == 2691826 and len(timing['rows']) == 1
due = datetime.fromisoformat(timing['rows'][0]['estimated_training_completion_at']) - timedelta(minutes=5)
out = private/'final_report_observer761'
assert not out.exists()
out.mkdir()
record = {'status':'WAITING_ORIGINAL_REPORT_MILESTONE','pid':os.getpid(),
          'due':due.isoformat(),'poll_seconds':240,'controller_pid':2691826,
          'boundary':'Read-only original controller observation; no report/model/scoring/training invocation or restart.'}

def save():
    record['updated_at'] = datetime.now().astimezone().isoformat()
    (out/'STATUS.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')

save()
while datetime.now().astimezone() < due:
    time.sleep(min(240,(due-datetime.now().astimezone()).total_seconds()))
index = 0
while True:
    index += 1
    label = f'final_report761_{index:04d}'
    result = subprocess.run([sys.executable,'-X','utf8',
                             'C:/Users/gb/.codex_tmp/observe_metric_scale755.py','--label',label],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out/f'{label}.stdout.txt').write_bytes(result.stdout)
    (out/f'{label}.stderr.txt').write_bytes(result.stderr)
    record.update(last_label=label,last_observer_exit=result.returncode)
    if result.returncode:
        record['status'] = 'OBSERVATION_FAILED_NO_RESTART'
        save()
        raise SystemExit(result.returncode)
    path = private/'observations'/label/'STATUS.json'
    actual = json.loads(path.read_bytes())
    state = actual['files']['campaign.json']
    full = [row for row in state['jobs'] if row['phase'] == 'full']
    complete = [row for row in full if row['status'] == 'COMPLETE' and row['exit_code'] == 0]
    record.update(last_receipt=str(path),actual_campaign_status=state['status'],
                  completed_formal_jobs=len(complete),controller_live=actual['controller_live'],
                  report_invocations=state['report_invocations'],report_exit_code=state.get('report_exit_code'))
    if (len(complete) == 6 and state['status'] == 'COMPLETE'
            and state['report_invocations'] == 1 and state.get('report_exit_code') == 0
            and not actual['controller_live']):
        record['status'] = 'ORIGINAL_ALL6_AND_ONCE_REPORT_COMPLETE_OBSERVED'
        save()
        break
    if state['status'] == 'FAILED' or not actual['controller_live']:
        record['status'] = 'CONTROLLER_TERMINAL_WITHOUT_VERIFIED_REPORT'
        save()
        break
    record['status'] = 'WAITING_ORIGINAL_ALL6_AND_ONCE_REPORT_COMPLETE'
    save()
    time.sleep(240)
print(json.dumps(record, indent=2))
