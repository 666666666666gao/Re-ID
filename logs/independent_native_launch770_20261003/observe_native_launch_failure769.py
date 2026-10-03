from datetime import datetime
from pathlib import Path
import json
import paramiko

target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/launch_failure_resources769')
assert not target.exists()
target.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
code = '''from pathlib import Path
from datetime import datetime
import json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
paths=sorted(Path('/tmp').glob('trifusion_target*_20261003.bundle'))
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'disk_free_bytes':shutil.disk_usage(root).free,'required_bytes':10292822016,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),'head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'new_campaign_exists':(root/'logs/independent_native_evidence_20261003_v1').exists(),'new_launchlog_exists':(root/'logs/independent_native_evidence_20261003_v1.launcher.log').exists(),'transport_bundles':[{'path':str(p),'bytes':p.stat().st_size,'resolved':str(p.resolve()),'symlink':p.is_symlink(),'filesystem_matches_project':p.stat().st_dev==root.stat().st_dev} for p in paths],'boundary':'Read-only; no launch or retirement.'}))
'''
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(target / 'stdout.json').write_bytes(data)
(target / 'stderr.txt').write_bytes(error)
(target / 'EXIT.json').write_text(json.dumps({'exit_code': exit_code, 'at': datetime.now().astimezone().isoformat()}) + '\n')
client.close()
assert exit_code == 0, error.decode()
print(data.decode(), end='')
