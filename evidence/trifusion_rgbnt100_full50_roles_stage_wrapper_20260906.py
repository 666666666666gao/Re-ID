from pathlib import Path
import argparse
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
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('m0', 'main'), required=True)
args = parser.parse_args()
stage = 'roles_' + args.mode
registration = ROOT / 'evidence/trifusion_rgbnt100_full50_roles_registration_20260906.json'
reg = json.loads(registration.read_text())
config = ROOT / reg['config']
for path, expected in [(config, reg['config_sha256']), (ROOT / reg['runner'], reg['runner_sha256']),
                       (ROOT / reg['verifier'], reg['verifier_sha256'])]:
    assert sha(path) == expected, str(path)
RUN.mkdir(exist_ok=True)
assert not (RUN / stage).exists() and not (RUN / (stage + '_launch.json')).exists()
assert shutil.disk_usage(RUN).free >= 8 * 1024**3
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,memory.total',
                               '--format=csv,noheader,nounits'], text=True).strip()
gpu_row = [int(x.strip()) for x in gpu.split(',')]
assert len(gpu_row) == 4 and gpu_row[1] < 500 and gpu_row[2] >= 22000, gpu
command = [PY, '-u', str(ROOT / reg['runner']), '--config', str(config), '--mode', args.mode,
           '--output-dir', str(RUN / stage)]
verify = [PY, '-u', str(ROOT / reg['verifier']), '--config', str(config), '--summary',
          str(RUN / stage / 'summary.json'), '--output', str(RUN / (stage + '_verification.json'))]
if args.mode == 'main':
    command += ['--m0-receipt', str(RUN / 'roles_m0/summary.json'),
                '--m0-verification', str(RUN / 'roles_m0_verification.json')]
    verify += ['--m0-receipt', str(RUN / 'roles_m0/summary.json')]
launch = {'launched_at': datetime.datetime.now().astimezone().isoformat(), 'wrapper_pid': os.getpid(),
          'mode': args.mode, 'execution_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'command': command, 'verification_command': verify, 'config_sha256': sha(config),
          'registration_sha256': sha(registration), 'wrapper_sha256': sha(Path(__file__)),
          'gpu_prelaunch': gpu, 'output_volume_free_bytes': shutil.disk_usage(RUN).free,
          'official_model_record_forwards': 0, 'rgbnt201_dev_record_forwards': 0}
(RUN / (stage + '_launch.json')).write_text(json.dumps(launch, indent=2) + '\n')
started = time.perf_counter()
with (RUN / (stage + '.log')).open('w') as log:
    training_code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
(RUN / (stage + '_training_exit.txt')).write_text(str(training_code) + '\n')
verification_code = None
if training_code == 0:
    with (RUN / (stage + '_verification.log')).open('w') as log:
        verification_code = subprocess.run(verify, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
code = training_code if training_code != 0 else verification_code
(RUN / (stage + '_exit.txt')).write_text(str(code) + '\n')
(RUN / (stage + '_terminal.json')).write_text(json.dumps({**launch,
    'completed_at': datetime.datetime.now().astimezone().isoformat(), 'training_exit_code': training_code,
    'verification_exit_code': verification_code, 'exit_code': code,
    'elapsed_seconds': time.perf_counter() - started, 'output_volume_free_bytes_after': shutil.disk_usage(RUN).free}, indent=2) + '\n')
sys.exit(code)
