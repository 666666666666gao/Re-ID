from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
source = root / '.codex_tmp/launch_patch_memory_100_diagnosis_20261001.py'
receipt = root / 'logs/patch_memory_100_diagnosis_launch_20261001.json'
log = root / '.codex_tmp/patch_memory_100_diagnosis_20261001.log'
assert not receipt.exists() and not log.exists()
compile(source.read_text(), str(source), 'exec')
with log.open('x') as output:
    process = subprocess.Popen([sys.executable, '-B', str(source)], cwd=root,
                               stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
record = {'launched_at': datetime.now().astimezone().isoformat(), 'pid': process.pid,
          'command': [sys.executable, '-B', str(source)], 'log': str(log),
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'scope': 'Unchanged RGBNT100 CPU/GPU diagnostics; no optimizer or training'}
receipt.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
