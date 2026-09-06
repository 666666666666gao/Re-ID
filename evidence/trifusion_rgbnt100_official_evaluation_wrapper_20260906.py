from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN = Path('/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_trifusion_main_v1_seed42_20260906')
PY = '/root/miniconda3/envs/tri_reid/bin/python'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
registration = ROOT / 'evidence/trifusion_rgbnt100_official_evaluation_registration_20260906.json'
reg = json.loads(registration.read_text())
config = ROOT / reg['config']
for path, expected in [(config, reg['config_sha256']), (ROOT / reg['runner'], reg['runner_sha256']),
                       (ROOT / reg['verifier'], reg['verifier_sha256'])]:
    assert sha(path) == expected, str(path)
assert not (RUN / 'official').exists() and not (RUN / 'official_launch.json').exists()
assert shutil.disk_usage(RUN).free >= 8 * 1024**3
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,memory.total',
                               '--format=csv,noheader,nounits'], text=True).strip()
gpu_row = [int(x.strip()) for x in gpu.split(',')]
assert len(gpu_row) == 4 and gpu_row[1] < 500 and gpu_row[2] >= 22000, gpu
command = [PY, '-u', str(ROOT / reg['runner']), '--config', str(config), '--output-dir', str(RUN / 'official')]
verify = [PY, '-u', str(ROOT / reg['verifier']), '--config', str(config), '--summary',
          str(RUN / 'official/summary.json'), '--output', str(RUN / 'official_verification.json')]
launch = {'launched_at': datetime.datetime.now().astimezone().isoformat(), 'wrapper_pid': os.getpid(),
          'execution_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'command': command, 'verification_command': verify, 'config_sha256': sha(config),
          'registration_sha256': sha(registration), 'wrapper_sha256': sha(Path(__file__)),
          'gpu_prelaunch': gpu, 'output_volume_free_bytes': shutil.disk_usage(RUN).free,
          'fixed_model_epochs': {'signal': 30, 'roles': 20}, 'expected_official_record_forwards_per_model': 10290}
(RUN / 'official_launch.json').write_text(json.dumps(launch, indent=2) + '\n')
started = time.perf_counter()
with (RUN / 'official.log').open('w') as log:
    evaluation_code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
(RUN / 'official_evaluation_exit.txt').write_text(str(evaluation_code) + '\n')
verification_code = None
if evaluation_code == 0:
    with (RUN / 'official_verification.log').open('w') as log:
        verification_code = subprocess.run(verify, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
code = evaluation_code if evaluation_code != 0 else verification_code
(RUN / 'official_exit.txt').write_text(str(code) + '\n')
(RUN / 'official_terminal.json').write_text(json.dumps({**launch,
    'completed_at': datetime.datetime.now().astimezone().isoformat(), 'evaluation_exit_code': evaluation_code,
    'verification_exit_code': verification_code, 'exit_code': code,
    'elapsed_seconds': time.perf_counter() - started, 'output_volume_free_bytes_after': shutil.disk_usage(RUN).free}, indent=2) + '\n')
sys.exit(code)
