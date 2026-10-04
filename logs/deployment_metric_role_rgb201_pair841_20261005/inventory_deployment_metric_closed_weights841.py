"""Read own weight metadata after observed disk decline; do not delete anything."""
from datetime import datetime
from pathlib import Path
import json
import paramiko

packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/deployment_metric_weight_inventory841')
assert not packet.exists()
code = '''from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
folder=root/'trained-model'
seal=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
protected=set(seal['artifact_sha256'])
rows=[]
for path in sorted(folder.rglob('*.pth')):
 rows.append(dict(path=str(path),bytes=path.stat().st_size,protected_current_control=str(path) in protected,
                  sibling_texts=sorted(p.name for p in path.parent.glob('*.json'))))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),disk_free_bytes=shutil.disk_usage(root).free,
                     weights=rows,total_weight_bytes=sum(r['bytes'] for r in rows),
                     boundary='Own trained-model weight file metadata only,not a deletion eligibility judgment. No model/GPU/progress query or power/temperature action.')))
'''
compile(code, 'remote_owned_weight_inventory841.py', 'exec')
packet.mkdir()
(packet / 'remote_source.py').write_text(code, encoding='utf-8')
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, out, err = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
out.channel.settimeout(180)
data, error = out.read(), err.read()
rc = out.channel.recv_exit_status()
client.close()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat())) + '\n')
assert rc == 0, error.decode()
record = json.loads(data)
print(json.dumps(dict(at=record['at'],disk_free_bytes=record['disk_free_bytes'],weights=len(record['weights']),total_weight_bytes=record['total_weight_bytes'],
                     non_best_candidates=[r for r in record['weights'] if r['path'].rsplit('/',1)[-1]!='best_map.pth' and not r['protected_current_control']])))
