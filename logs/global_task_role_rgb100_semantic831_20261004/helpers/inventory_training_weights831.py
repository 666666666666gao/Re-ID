"""Read current project checkpoint sizes after the observed free-space decrease."""
from datetime import datetime
import json
from pathlib import Path
import paramiko

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/training_weight_inventory831')
assert not packet.exists()
packet.mkdir()
code='''from pathlib import Path
from datetime import datetime
import json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
folder=root/'trained-model'
names=subprocess.check_output(['rg','--files',str(folder),'-g','*.pth'],text=True).splitlines()
rows=[{'path':name,'bytes':Path(name).stat().st_size} for name in names]
sizes=subprocess.check_output(['du','-sx','--block-size=1',str(folder),str(root/'logs'),str(root/'results'),str(root/'.git')],text=True)
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'disk_free_bytes':shutil.disk_usage(root).free,'checkpoints':rows,'directory_bytes':sizes,'boundary':'Read-only own trained-model checkpoint metadata and own project directory sizes. No model/GPU/temperature/power access, deletion or current-source modification.'}))
'''
(packet/'remote_inventory.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(60)
data,error=stdout.read(),stderr.read()
status=stdout.channel.recv_exit_status()
client.close()
(packet/'stdout.json').write_bytes(data)
(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status,at=datetime.now().astimezone().isoformat()))+'\n')
assert status==0,error.decode()
r=json.loads(data)
print(json.dumps(dict(at=r['at'],disk_free_bytes=r['disk_free_bytes'],checkpoint_count=len(r['checkpoints']),checkpoint_bytes=sum(x['bytes'] for x in r['checkpoints']),probe_checkpoints=[x for x in r['checkpoints'] if 'probe' in Path(x['path']).name],directory_bytes=r['directory_bytes']),indent=2))
