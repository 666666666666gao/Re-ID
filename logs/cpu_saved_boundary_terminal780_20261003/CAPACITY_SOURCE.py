from datetime import datetime
from pathlib import Path
import json
import paramiko

target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/capacity780')
assert not target.exists()
target.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
code = '''from datetime import datetime
from pathlib import Path
import json, shutil, subprocess
query = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, check=True)
topology = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True, check=True)
print(json.dumps({'at': datetime.now().astimezone().isoformat(),
 'gpu_inventory': query.stdout, 'topology': topology.stdout,
 'ram': {line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))},
 'free_bytes': {path: shutil.disk_usage(path).free for path in ('/data/gaob/Re-ID/Trifusion', '/home/gaob')},
 'boundary':'2026 read-only resource inventory; no processes inspected, model import, CUDA workload, training or cleanup.'}, indent=2))
'''
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
rc = stdout.channel.recv_exit_status()
client.close()
(target/'stdout.json').write_bytes(data)
(target/'stderr.txt').write_bytes(error)
(target/'EXIT.json').write_text(json.dumps({'exit_code':rc, 'at':datetime.now().astimezone().isoformat()})+'\n')
assert rc == 0, error.decode()
print(data.decode())
