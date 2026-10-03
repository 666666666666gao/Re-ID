"""Estimate from actual elapsed epoch progress; arm no training or scoring."""
from pathlib import Path
from datetime import datetime, timedelta
import json

private = Path('C:/Users/gb/.codex_tmp/training_feature_scale_split755')
source = private/'observations/full_start757/STATUS.json'
actual = json.loads(source.read_bytes())
now = datetime.fromisoformat(actual['observed_at'])
rows = []
for child in actual['children']:
    if child['phase'] != 'full':
        continue
    training = child['training']
    start = datetime.fromisoformat(training['started_at'])
    complete = len(training['history'])
    elapsed = (now-start).total_seconds()
    assert 0 < complete < 50
    lower = start+timedelta(seconds=elapsed*50/(complete+1))
    upper = start+timedelta(seconds=elapsed*50/complete)
    rows.append({'dataset': child['dataset'], 'variant': child['variant'],
                 'training_started_at': start.isoformat(), 'completed_epochs': complete,
                 'last_epoch': child['last_step']['epoch'], 'last_batch': child['last_step']['batch'],
                 'recorded_steps': child['step_rows'], 'elapsed_wall_seconds': elapsed,
                 'estimated_training_end_lower': lower.isoformat(),
                 'estimated_training_end_upper': upper.isoformat(),
                 'last_loss': child['last_step']['loss']})
record = {'status': 'LIVE_PROGRESS_ETA_ONLY', 'observed_at': actual['observed_at'], 'rows': rows,
          'controller_pid': actual['controller_pid'], 'free_bytes': actual['free_bytes'],
          'next_observation_due': '2026-10-03T08:14:00+08:00', 'poll_seconds': 240,
          'boundary': 'Elapsed wall time includes epoch evaluation/save and the incomplete current epoch; bounds assume unchanged speed and are estimates, not completion evidence. Excludes final strict reload, queue recognition and final CPU report. history.seconds is training-loop-only and is not summed for ETA.'}
output = private/'LIVE_PROGRESS_ETA757.json'
assert not output.exists()
output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))
