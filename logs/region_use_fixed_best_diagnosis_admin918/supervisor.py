"""Observe one inference-only controller; preserve its true exit."""
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime

root = Path('/data/gaob/Re-ID/Trifusion')
admin = root / 'logs/region_use_fixed_best_diagnosis_admin918'
assert not (admin / 'LAUNCH.json').exists() and not (admin / 'EXIT.json').exists()
command = ['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', str(root / 'tools/queue_region_use_best_diagnosis.py'),
           '--seal', str(root / 'refine-logs/region_use_fixed_best_diagnosis918/INPUT_SEAL.json'),
           '--output-dir', str(root / 'logs/region_use_fixed_best_diagnosis_20261013_918')]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', PYTHONDONTWRITEBYTECODE='1')
with (admin / 'controller.log').open('x') as log:
    process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    launch = dict(status='READ_ONLY_INFERENCE_CONTROLLER_LAUNCHED', started_at=datetime.now().astimezone().isoformat(),
                  supervisor_pid=os.getpid(), controller_pid=process.pid, command=command,
                  physical_gpus=[0,1], optimizer_updates=0, training=False)
    (admin / 'LAUNCH.json').write_text(json.dumps(launch, indent=2) + '\n')
    code = process.wait()
terminal = dict(status='CONTROLLER_EXIT_OBSERVED', completed_at=datetime.now().astimezone().isoformat(),
                supervisor_pid=os.getpid(), controller_pid=process.pid, exit_code=code, optimizer_updates=0, training=False)
(admin / 'EXIT.json').write_text(json.dumps(terminal, indent=2) + '\n')
raise SystemExit(code)
