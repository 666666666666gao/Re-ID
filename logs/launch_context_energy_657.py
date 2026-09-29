from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
output = root / '.git/context_energy_launch_657_20260929.json'
assert not output.exists()
source = root / 'tools/diagnose_correspondence_context_energy.py'
assert sha(source) == 'dba46b8f73460191318e7ba5d42a4bb191409b3ed639d5bf6eb1f7aff3ffb8e6'
campaign = root / 'logs/correspondence_context_identity_20260929'
manifest = json.loads((campaign / 'manifest.json').read_text())
assert all(sha(root / name) == digest for name, digest in manifest['source_sha256'].items())
state = json.loads((campaign / 'campaign.json').read_text())
assert state['status'] in ('RUNNING', 'COMPLETE')
assert all(row['status'] in ('RUNNING', 'COMPLETE') for row in state['jobs'])
assert json.loads((Path(manifest['after_campaign']) / 'campaign.json').read_text())['status'] == 'COMPLETE'
devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
used = {int(index): int(memory) for index, memory in (line.split(',') for line in devices.splitlines())}
assert used[1] < 500 and used[2] < 500
matrix = root / '.git/context_identity_accepted_667_20260929.json'
records = json.loads(matrix.read_text())
assert records['verified_complete'] == 13
jobs = []
for gpu, dataset in ((1, 'RGBNT201'), (2, 'MSVR310')):
    folder = root / f'.git/context_energy_{dataset}_657_20260929'
    assert not folder.exists()
    command = [sys.executable, '-B', str(source), '--matrix', str(matrix),
               '--dataset', dataset, '--output-dir', str(folder)]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    log = root / f'.git/context_energy_{dataset}_657_20260929.log'
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    jobs.append({'gpu': gpu, 'dataset': dataset, 'pid': child.pid,
                 'command': command, 'output_dir': str(folder), 'log': str(log)})
report = {'status': 'READ_ONLY_ENERGY_DIAGNOSTICS_STARTED',
          'at': datetime.now().astimezone().isoformat(), 'source_sha256': sha(source),
          'matrix_sha256': sha(matrix), 'gpu_memory_before': used,
          'runtime_source_sha256': manifest['source_sha256'], 'jobs': jobs,
          'boundary': 'Ten predeclared accepted best checkpoints, normal full query/gallery only. No training, checkpoint/weight search or overwrite of official outputs.'}
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
