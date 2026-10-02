"""Read the omitted report helper; preserve the original F1 binding gap."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import paramiko

BASE = Path('C:/Users/gb/.codex_tmp/foundation_recipe_complete_20261002')
OUT = BASE / 'diagnostic_source_closure'
assert not OUT.exists()
name = 'tools/analyze_correspondence_distances.py'
prior = Path('C:/Users/gb/.codex_tmp/semantic_native_complete734_20261002')
prior_receipt = json.loads((prior / 'collection.json').read_text(encoding='utf-8'))
prior_source = (prior / 'raw' / name).read_bytes()
digest = hashlib.sha256(prior_source).hexdigest()
assert prior_receipt['files'][name] == digest == prior_receipt['waiter']['analyzer_source_sha256']
assert datetime.fromisoformat(prior_receipt['collected_at']) < datetime.fromisoformat('2026-10-02T19:19:55+08:00')
git_source = subprocess.check_output(['git', 'show', '8ca21160caaab9c7c96acbd0504ed8d39be7d883:' + name],
                                    cwd='C:/Users/gb/.trifusion_github_publish_22c3bee')
assert hashlib.sha256(git_source).hexdigest() == digest
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2025, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
remote = '/data2/gb/Re-ID/Trifusion/' + name
code = f"from pathlib import Path;import hashlib,json;from datetime import datetime;p=Path({remote!r});print(json.dumps({{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mtime':datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(),'observed_at':datetime.now().astimezone().isoformat()}}))"
_, stdout, stderr = client.exec_command(shlex.join(['/usr/bin/python3', '-B', '-c', code]))
raw, error = stdout.read(), stderr.read()
assert stdout.channel.recv_exit_status() == 0, error.decode()
receipt = json.loads(raw)
sftp = client.open_sftp()
data = sftp.open(remote, 'rb').read()
sftp.close()
client.close()
assert len(data) == receipt['bytes'] and hashlib.sha256(data).hexdigest() == receipt['sha256'] == digest
original = json.loads((BASE / 'INTAKE.json').read_text(encoding='utf-8'))
assert name not in original['source_sha256']
OUT.mkdir()
(OUT / name).parent.mkdir(parents=True)
(OUT / name).write_bytes(data)
(OUT / 'prior734_collection.json').write_bytes((prior / 'collection.json').read_bytes())
receipt.update(prior734_collected_at=prior_receipt['collected_at'],
               prior734_source_sha256=digest, committed_8ca_source_sha256=digest,
               original_f1_source249_includes_helper=False,
               original_f1_manifest_modified=False,
               scope='Current2025 source equals pre-F1 primary734 source and committed8ca source; omitted from F1 frozen249. This post-execution intake does not retrospectively add a runtime immutability witness or repair the original manifest. No code/report/scorer execution or result change.')
(OUT / 'SOURCE_CLOSURE.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt, indent=2))
