"""Verify the completed publication without claiming the unavailable 2025 mirror."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import subprocess
import urllib.request
import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
target = proof / 'four_copy822_2025_pending.json'
assert not target.exists() and not (proof / 'five_copy822.json').exists()
assert json.loads((base / 'publication822_attempt/EXIT.json').read_bytes())['exit_code'] == 1
failure = (base / 'publication822_attempt/stderr.txt').read_text(encoding='utf-8')
assert 'Input/output error' in failure and '/data2/gb/' in failure
check = json.loads((base / 'publication822_mirror_io_check/CHECK.json').read_bytes())
assert all(row['exit_code'] == 1 and 'Input/output error' in row['stderr'] for row in check['rows'] if row['port'] == 2025)
head = json.loads((proof / 'commit822.json').read_bytes())['head']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == head
assert subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=repo) == b''
files = sorted({name for number in range(739, 823) for name in json.loads((proof / f'publication{number}_local.json').read_bytes())['files']})
blob = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=repo, input=''.join(head + ':' + name + '\n' for name in files).encode())
offset = 0
expected = {}
for name in files:
    end = blob.index(b'\n', offset)
    size = int(blob[offset:end].split()[-1])
    offset = end + 1
    data = blob[offset:offset + size]
    assert data == (repo / name).read_bytes(), name
    expected[name] = hashlib.sha256(data).hexdigest()
    offset += size + 1
doc = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
assert hashlib.sha256(urllib.request.urlopen('https://raw.githubusercontent.com/666666666666gao/Re-ID/' + head + '/' + doc, timeout=30).read()).hexdigest() == expected[doc]
assert hashlib.sha256((Path('C:/Users/gb/Desktop/document') / Path(doc).name).read_bytes()).hexdigest() == expected[doc]
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
code = f"from pathlib import Path;import hashlib,json,subprocess;root=Path('/data/gaob/Re-ID/Trifusion');assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=={head!r};expected={expected!r};actual={{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected}};assert actual==expected;print(json.dumps(actual))"
stdin, out, err = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
out.channel.settimeout(60)
data, error = out.read(), err.read()
assert out.channel.recv_exit_status() == 0, error.decode()
client.close()
record = {'status': 'FOUR_COPIES_VERIFIED_2025_MIRROR_IO_PENDING', 'head': head, 'section': '41.822',
          'completed_at': datetime.now().astimezone().isoformat(), 'doc_sha256': expected[doc],
          'owned_count': len(files), 'servers': [{'port': 2026, 'verified_files': json.loads(data), 'text_sync_only': True}],
          'pending_mirror': {'port': 2025, 'root': '/data2/gb/Re-ID/Trifusion', 'status': 'INPUT_OUTPUT_ERROR',
                             'last_verified_section': '41.821', 'failure_receipt_sha256': hashlib.sha256((base / 'publication822_mirror_io_check/CHECK.json').read_bytes()).hexdigest()},
          'boundary': 'Local repository, Desktop, GitHub document and execution server verified. 2025 mirror not readable and not declared synchronized. No compute contract or science source changed; diagnostic execution only uses 2026 GPU0/1.'}
target.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: value for key, value in record.items() if key != 'servers'}))
