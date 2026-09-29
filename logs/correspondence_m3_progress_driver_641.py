from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root / 'tools'))
import queue_correspondence_refinement as queue

campaign = root / 'logs/correspondence_m3_prediction_20260929'
manifest = json.loads((campaign / 'manifest.json').read_text())
state = json.loads((campaign / 'campaign.json').read_text())
assert len(manifest['jobs']) == len(state['jobs']) == 12
assert manifest['poll_seconds'] == 240 and manifest['epochs'] == 50 and manifest['seed'] == 42
assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in manifest['source_sha256'].items())
rows = []
for job in state['jobs']:
    row = {key: job[key] for key in ('dataset', 'variant', 'status')}
    if job['status'] in ('RUNNING', 'COMPLETE', 'FAILED'):
        child = queue.child_campaign(campaign, 'm3', job['dataset'], job['variant'])
        child_state = json.loads((child / 'campaign.json').read_text())
        phase = child_state['jobs'][-1]
        row.update(gpu=job['gpu'], worker_pid=job['pid'], child_status=child_state['status'], mode=phase['mode'],
                   phase_status=phase['status'], pid=phase['pid'], output_dir=phase['output_dir'])
        process = Path(f"/proc/{phase['pid']}/cmdline")
        row['actual_command'] = process.read_bytes().replace(b'\0', b' ').decode() if process.exists() else None
        record = json.loads((Path(phase['output_dir']) / ('official_metrics.json' if phase['mode'] == 'evaluate' else 'training.json')).read_text())
        row.update(receipt_status=record['status'], completed_epochs=len(record.get('history', [])))
        m0_record = json.loads((Path(child_state['jobs'][0]['output_dir']) / 'training.json').read_text())
        row['m0_status'] = m0_record['status']
        if m0_record['status'] == 'M0_PASS':
            row['m0'] = m0_record['m0']
        if record.get('history'):
            row['latest_epoch_seconds'] = record['history'][-1]['seconds']
            row['latest_steps'] = record['history'][-1]['steps']
    rows.append(row)
controller = Path(f"/proc/{state['controller_pid']}/cmdline")
report = {'status': state['status'], 'at': datetime.now().astimezone().isoformat(),
          'campaign': str(campaign), 'manifest_sha256': hashlib.sha256((campaign / 'manifest.json').read_bytes()).hexdigest(),
          'controller_pid': state['controller_pid'],
          'controller_command': controller.read_bytes().replace(b'\0', b' ').decode() if controller.exists() else None,
          'jobs': rows, 'data_free_bytes': shutil.disk_usage(root).free,
          'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True),
          'gpu_processes': subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True),
          'boundary': 'Actual process and saved phase/epoch observation only; no intermediate best promoted to formal result. Frozen eight-source manifest checked without changing scientific code.'}
out = root / '.git/correspondence_m3_progress_641_20260929.json'
assert not out.exists()
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
