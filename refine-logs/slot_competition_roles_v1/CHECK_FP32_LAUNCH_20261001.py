from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/data/gaob/Re-ID/Trifusion')
campaign = root / 'logs/slot_competition_fp32_roles_20261001_r2'
target = root / 'logs/slot_competition_fp32_first_wave_20261001_r2.json'
assert not target.exists()
state = json.loads((campaign / 'campaign.json').read_text())
manifest = json.loads((campaign / 'manifest.json').read_text())
assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
           for name, digest in manifest['source_sha256'].items())
rows = []
for job in state['jobs']:
    row = {key: job.get(key) for key in ('dataset', 'variant', 'status', 'pid', 'gpu')}
    if job['status'] == 'RUNNING':
        folder = campaign / f"{campaign.name}_slot_competition_fp32_{job['variant']}_{job['dataset']}"
        child = json.loads((folder / 'campaign.json').read_text())
        row['child_status'] = child['status']
        row['child_jobs'] = [{key: stage.get(key) for key in ('mode', 'status', 'pid', 'exit_code')}
                             for stage in child['jobs']]
        for stage in child['jobs']:
            if stage['mode'] == 'm0' and stage['status'] == 'COMPLETE':
                receipt = json.loads((Path(stage['output_dir']) / 'training.json').read_text())
                assert receipt['status'] == 'M0_PASS' and stage['exit_code'] == 0
                assert receipt['m0']['frozen_signal_unchanged']
                assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters']
                assert receipt['initializer']['attention_subgraph_dtype'] == 'float32'
                row['m0'] = receipt['m0']
                row['initializer'] = receipt['initializer']
            if stage['mode'] == 'train':
                receipt = json.loads((Path(stage['output_dir']) / 'training.json').read_text())
                row['completed_epochs'] = len(receipt['history'])
                row['first_epoch_seconds'] = receipt['history'][0]['seconds'] if receipt['history'] else None
    rows.append(row)
result = {'observed_at': datetime.now().astimezone().isoformat(), 'controller_status': state['status'],
          'controller_pid': state['controller_pid'], 'source_files_verified': len(manifest['source_sha256']),
          'jobs': rows, 'gpu_snapshot': subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], text=True),
          'scope': 'First real production M0 milestone only; no partial official metrics promoted'}
target.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({**{key: result[key] for key in ('observed_at', 'controller_status', 'source_files_verified', 'gpu_snapshot')},
                  'jobs': [{key: row.get(key) for key in ('dataset', 'variant', 'gpu', 'status', 'm0', 'completed_epochs', 'first_epoch_seconds')} for row in rows]}, indent=2))
