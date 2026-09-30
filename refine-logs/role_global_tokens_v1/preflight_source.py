"""Record actual CPU structural/help checks before any production run."""

import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
OUT = ROOT / 'refine-logs/role_global_tokens_v1/PREFLIGHT_20261001.json'
assert not OUT.exists()
names = ('modeling/trifusion/role_global_tokens.py', 'tools/run_role_global_tokens.py',
         'tools/check_role_global_tokens_cpu.py', 'tools/queue_role_global_tokens.py',
         'tools/collect_role_global_tokens.py', 'refine-logs/role_global_tokens_v1/EXPERIMENT_PLAN.md')
sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
for name in names[:-1]:
    ast.parse((ROOT / name).read_text())
previous = ('patch_memory_roles_recovery_20261001', 'slot_competition_roles_20261001',
            'slot_competition_fp32_roles_20261001_r2')
for campaign in previous:
    manifest = json.loads((ROOT / 'logs' / campaign / 'manifest.json').read_text())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in manifest['source_sha256'].items())
commands = [
    [sys.executable, '-B', str(ROOT / 'tools/check_role_global_tokens_cpu.py')],
    [sys.executable, '-B', str(ROOT / 'tools/run_role_global_tokens.py'), '--token-mode', 'static',
     '--query-mode', 'context', '--auxiliary-target', 'none', '--help'],
    [sys.executable, '-B', str(ROOT / 'tools/queue_role_global_tokens.py'), '--help'],
    [sys.executable, '-B', str(ROOT / 'tools/collect_role_global_tokens.py'), '--help'],
]
record = {'started_at': datetime.now().astimezone().isoformat(), 'source_sha256': sources,
          'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'syntax_pass': True, 'prior_source_counts_verified': [210, 211, 213], 'checks': [],
          'scope': 'CPU structural fixture with linear Mamba stub and CLI help; no real data/GPU M0/training.'}
for index, command in enumerate(commands):
    log = OUT.parent / f'preflight_{index}.log'
    assert not log.exists()
    with log.open('x') as handle:
        child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        code = child.wait()
    record['checks'].append({'command': command, 'pid': child.pid, 'exit_code': code,
                            'log': str(log), 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
    record['status'] = 'FAILED' if code else 'RUNNING'
    OUT.write_text(json.dumps(record, indent=2) + '\n')
    if code:
        raise SystemExit(code)
record.update(status='CPU_AND_CLI_PASS', completed_at=datetime.now().astimezone().isoformat())
OUT.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
