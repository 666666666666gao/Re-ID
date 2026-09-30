"""Run unchanged full-query pair and slot diagnostics after all six acceptances."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
matrix = root / 'logs/patch_memory_roles_recovery_20261001/accepted_matrix.json'
target = root / 'logs/patch_memory_100_diagnosis_20261001'
accepted = json.loads(matrix.read_text())
assert accepted['verified_complete'] == accepted['expected_endpoints'] == 6
assert all(row['status'] == 'VERIFIED_COMPLETE' for row in accepted['rows'])
assert not target.exists()
occupied = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True)
assert not occupied.strip(), occupied
slot = root / 'tools/diagnose_patch_memory_slots.py'
assert hashlib.sha256(slot.read_bytes()).hexdigest() == 'af6c92ab77dd702b5e1b4eae46eb099889952fb01b66929c41d8e9ff955e2b32'
manifest = json.loads((matrix.parent / 'manifest.json').read_text())
assert len(manifest['source_sha256']) == 210
assert all(hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
           for path, digest in manifest['source_sha256'].items())
target.mkdir()
(target / 'cpu').mkdir()
jobs = [
    {'name': 'cpu', 'gpu': '', 'command': [sys.executable, '-B',
     str(root / 'tools/analyze_correspondence_distances.py'), '--matrix', str(matrix),
     '--dataset', 'RGBNT100', '--control', 'local_memory', '--candidate', 'full_memory',
     '--output', str(target / 'cpu/RGBNT100.json')]},
    {'name': 'slots', 'gpu': '1', 'command': [sys.executable, '-B', str(slot),
     '--matrix', str(matrix), '--dataset', 'RGBNT100', '--output-dir', str(target / 'slots')]},
]
report = {'status': 'RUNNING', 'started_at': datetime.now().astimezone().isoformat(),
          'controller_pid': os.getpid(), 'matrix_sha256': hashlib.sha256(matrix.read_bytes()).hexdigest(),
          'source_sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (slot, root / 'tools/analyze_correspondence_distances.py')},
          'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'jobs': jobs}


def save():
    (target / 'campaign.json').write_text(json.dumps(report, indent=2) + '\n')


processes = []
for job in jobs:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = job['gpu']
    log = (target / (job['name'] + '.log')).open('x')
    process = subprocess.Popen(job['command'], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    job.update(pid=process.pid, status='RUNNING')
    processes.append(process)
    save()
for job, process in zip(jobs, processes):
    code = process.wait()
    job.update(exit_code=code, status='COMPLETE' if code == 0 else 'FAILED',
               waited_at=datetime.now().astimezone().isoformat())
    save()
report.update(status='COMPLETE' if all(job['exit_code'] == 0 for job in jobs) else 'FAILED',
              completed_at=datetime.now().astimezone().isoformat())
save()
print(json.dumps(report), flush=True)
sys.exit(0 if report['status'] == 'COMPLETE' else 1)
