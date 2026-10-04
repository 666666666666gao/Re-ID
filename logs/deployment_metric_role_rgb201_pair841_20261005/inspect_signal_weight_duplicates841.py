"""Hash only the observed historical Signalbest copies; no deletion."""
from datetime import datetime
from pathlib import Path
import json
import paramiko

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
inventory = json.loads((base / 'deployment_metric_weight_inventory841/stdout.json').read_bytes())
paths = [r['path'] for r in inventory['weights'] if r['path'].endswith('/Signalbest.pth')]
packet = base / 'signal_duplicate_weight_read841'
assert not packet.exists()
code = '''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');folder=root/'trained-model'
paths=PATHS
rows=[]
for name in paths:
 path=Path(name);assert path.resolve().is_relative_to(folder.resolve())
 stat=path.stat();digest=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
 rows.append(dict(path=name,bytes=stat.st_size,sha256=digest.hexdigest(),device=stat.st_dev,inode=stat.st_ino,links=stat.st_nlink))
groups={}
for row in rows:groups.setdefault(row['sha256'],[]).append(row)
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),disk_free_bytes=shutil.disk_usage(root).free,rows=rows,
                     duplicate_groups=[v for v in groups.values() if len(v)>1],
                     boundary='Observed historical immutable Signal weight copies only. No deletion,NN,GPU,power/temp or running-queue query.')))
'''.replace('paths=PATHS', 'paths=' + repr(paths))
compile(code, 'remote_signal_duplicate_read841.py', 'exec')
packet.mkdir()
(packet / 'remote_source.py').write_text(code, encoding='utf-8')
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, out, err = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
out.channel.settimeout(300)
data, error = out.read(), err.read()
rc = out.channel.recv_exit_status()
client.close()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat())) + '\n')
assert rc == 0, error.decode()
record = json.loads(data)
print(json.dumps(dict(at=record['at'],disk_free_bytes=record['disk_free_bytes'],copies=len(record['rows']),duplicate_groups=record['duplicate_groups'])))
