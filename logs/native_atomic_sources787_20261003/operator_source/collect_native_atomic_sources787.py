from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import hashlib
import json
import urllib.request

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
target = base / 'native_atomic_sources787'
assert not target.exists()
target.mkdir()
origin = 'https://raw.githubusercontent.com/state-spaces/mamba/v2.2.6.post3/'
names = [
    'mamba_ssm/modules/mamba_simple.py',
    'mamba_ssm/ops/selective_scan_interface.py',
    'csrc/selective_scan/selective_scan.cpp',
    'csrc/selective_scan/selective_scan_bwd_kernel.cuh',
    'csrc/selective_scan/selective_scan.h',
    'setup.py',
]
installed = base / 'native_installed_sources786_actual_python/source'


def fetch(name):
    url = origin + name
    with urllib.request.urlopen(url, timeout=30) as response:
        assert response.status == 200
        data = response.read()
        headers = dict(response.headers)
    path = target / 'upstream' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    row = {'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
           'at': datetime.now().astimezone().isoformat(), 'headers': headers}
    if name.startswith('mamba_ssm/'):
        current = (installed / name).read_bytes()
        row.update(installed_bytes=len(current), installed_sha256=hashlib.sha256(current).hexdigest(),
                   installed_python_byte_equal=current == data)
    return name, row


with ThreadPoolExecutor(max_workers=6) as pool:
    rows = dict(pool.map(fetch, names))
record = {'at': datetime.now().astimezone().isoformat(), 'upstream_version': 'v2.2.6.post3',
          'files': rows, 'boundary': 'Official versioned text only, compared to installed Python text. '
          'No Torch/model/CUDA import or execution, no installed compiled-binary provenance or runtime branch proof.'}
(target / 'INTAKE.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'at': record['at'], 'files': len(rows),
                  'python_byte_equal': {name: row['installed_python_byte_equal']
                                        for name, row in rows.items() if 'installed_python_byte_equal' in row}}))
