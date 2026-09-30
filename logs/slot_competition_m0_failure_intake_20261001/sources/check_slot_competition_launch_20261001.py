from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/data/gaob/Re-ID/Trifusion')
campaign = root / 'logs/slot_competition_roles_20261001'
target = root / 'logs/slot_competition_first_wave_20261001.json'
assert not target.exists()
state = json.loads((campaign / 'campaign.json').read_text())
manifest = json.loads((campaign / 'manifest.json').read_text())
assert state['status'] == 'RUNNING' and len(state['jobs']) == 6
assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
           for name, digest in manifest['source_sha256'].items())
rows = []
for job in state['jobs']:
    row = {key: job.get(key) for key in ('dataset', 'variant', 'status', 'pid', 'gpu')}
    if job['status'] == 'RUNNING':
        folder = campaign / f"{campaign.name}_slot_competition_{job['variant']}_{job['dataset']}"
        child = json.loads((folder / 'campaign.json').read_text())
        row['child_jobs'] = [{key: stage.get(key) for key in ('mode', 'status', 'pid', 'exit_code')}
                             for stage in child['jobs']]
        for stage in child['jobs']:
            if stage['mode'] == 'm0' and stage['status'] == 'COMPLETE':
                receipt = json.loads((Path(stage['output_dir']) / 'training.json').read_text())
                assert receipt['status'] == 'M0_PASS' and stage['exit_code'] == 0
                assert receipt['m0']['frozen_signal_unchanged']
                assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters']
                row['m0'] = receipt['m0']
                row['initializer'] = receipt['initializer']
            if stage['mode'] == 'train':
                receipt = json.loads((Path(stage['output_dir']) / 'training.json').read_text())
                row['completed_epochs'] = len(receipt['history'])
                row['first_epoch_seconds'] = receipt['history'][0]['seconds'] if receipt['history'] else None
    rows.append(row)
result = {'observed_at': datetime.now().astimezone().isoformat(), 'controller_pid': state['controller_pid'],
          'controller_cmdline': (Path('/proc') / str(state['controller_pid']) / 'cmdline').read_bytes().replace(b'\0', b' ').decode(),
          'source_files_verified': len(manifest['source_sha256']), 'jobs': rows,
          'gpu_snapshot': subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], text=True),
          'scope': 'First launch/M0 milestone only; no partial official metrics promoted'}
target.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
