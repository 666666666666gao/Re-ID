"""Read own current artifacts and project directory sizes, with no deletion."""
from pathlib import Path
from datetime import datetime
import json,paramiko
packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/global_task_role_artifact_inventory828')
assert not packet.exists();packet.mkdir()
code="""from pathlib import Path
from datetime import datetime
import json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion');folder=root/'trained-model'
rows=[{'path':str(p),'bytes':p.stat().st_size} for d in folder.glob('global_task_role_v1_20261004_824*') for p in d.rglob('*') if p.is_file() and p.stat().st_size>10485760]
sizes=subprocess.check_output(['du','-sx','--block-size=1',str(folder),str(root/'logs'),str(root/'results'),str(root/'.git')],text=True)
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'disk_free_bytes':shutil.disk_usage(root).free,'own_large_artifacts':rows,'directory_bytes':sizes,'boundary':'Read-only metadata for current artifacts and own project directories; no tensor/model/GPU, no delete or power/temperature access.'}))
"""
(packet/'remote_inventory.py').write_text(code,encoding='utf-8')
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(60)
data,error=o.read(),e.read();status=o.channel.recv_exit_status();c.close()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps({'exit_code':status,'at':datetime.now().astimezone().isoformat()})+'\n')
assert status==0,error.decode();print(data.decode())
