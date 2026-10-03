"""One read-only 2026 resource and retained checkpoint size observation."""
from datetime import datetime
from pathlib import Path
import json
import paramiko

target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/resources769')
assert not target.exists()
target.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
code = '''from pathlib import Path
from datetime import datetime
import json, shutil, subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
print(json.dumps({'at':datetime.now().astimezone().isoformat(),
'disk_free_bytes':shutil.disk_usage(root).free,
'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True),
'compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True),
'f3_checkpoint_bytes':{str(p):p.stat().st_size for p in root.glob('trained-model/metric_feature_scale_20261003_v1_*/*.pth')},
'f3_status':json.loads((root/'logs/metric_feature_scale_20261003_v1/campaign.json').read_text())['status'],
'boundary':'Read-only, no process control or model execution;2026 only.'}))
'''
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(target/'stdout.json').write_bytes(data)
(target/'stderr.txt').write_bytes(error)
(target/'EXIT.json').write_text(json.dumps({'exit_code':exit_code,
    'collected_at':datetime.now().astimezone().isoformat()})+'\n')
client.close()
assert exit_code == 0, error.decode()
value = json.loads(data)
print(json.dumps({k:v for k,v in value.items() if k != 'f3_checkpoint_bytes'},indent=2))
