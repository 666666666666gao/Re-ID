from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess
import time

root = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
run = Path('/root/trifusion-storage/artifacts/rgbnt100_trifusion_source_oof_v1_seed42_20260906')
summary_path = run / 'comparison/summary.json'
summary = json.loads(summary_path.read_text())
assert (run / 'comparison_exit.txt').read_text().strip() == '0'
assert json.loads((run / 'comparison_terminal.json').read_text())['exit_code'] == 0
assert summary['status'] in ('COMPLETE_COMPARISON_SUPPORT_PASS', 'COMPLETE_COMPARISON_SUPPORT_FAIL')
assert len(summary['folds']) == 3 and summary['heldout_record_forwards'] == 8675
assert not (run / 'comparison_terminal_files.json').exists()
assert not (run / 'comparison_verification_launch.json').exists()
registration = json.loads((root / 'evidence/trifusion_rgbnt100_original_roles_registration_20260906.json').read_text())
verifier = root / 'tools/verify_rgbnt100_trifusion_terminal_files.py'
verifier_sha = hashlib.sha256(verifier.read_bytes()).hexdigest()
assert verifier_sha == registration['source_and_verifier_sha256'][verifier.relative_to(root).as_posix()]
command = ['/root/miniconda3/envs/tri_reid/bin/python', '-u', str(verifier),
           '--run-root', str(run), '--project-root', str(root),
           '--config', str(root / registration['config'])]
launch = {'launched_at': datetime.now().astimezone().isoformat(),
          'wrapper_pid': os.getpid(), 'command': command,
          'verification_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
          'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'verifier_sha256': verifier_sha,
          'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
          'model_image_forwards': 0, 'optimizer_updates': 0, 'backward_calls': 0}
(run / 'comparison_verification_launch.json').write_text(json.dumps(launch, indent=2) + '\n')
started = time.perf_counter()
with (run / 'comparison_verification.log').open('w') as log:
    code = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT).returncode
(run / 'comparison_verification_exit.txt').write_text(str(code) + '\n')
terminal = {**launch, 'completed_at': datetime.now().astimezone().isoformat(),
            'exit_code': code, 'elapsed_seconds': time.perf_counter() - started}
(run / 'comparison_verification_terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
print(json.dumps(terminal))

