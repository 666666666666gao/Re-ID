"""Inventory own large engineering files outside model/data stores."""
from pathlib import Path
from datetime import datetime
import json
import paramiko
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft');packet=base/'engineering_file_inventory842';assert not packet.exists()
code='''from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');protect=set(json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256'])
rows=[dict(path=str(p),bytes=p.stat().st_size,protected_current=str(p) in protect) for folder in (root/'logs',root/'results') for p in folder.rglob('*') if p.is_file() and p.stat().st_size>=128*1024**2]
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),disk_free_bytes=shutil.disk_usage(root).free,rows=sorted(rows,key=lambda r:r['bytes'],reverse=True),boundary='Own logs/results filesystem metadata only,no deletion eligibility claim or model/power-temperature query.')))
'''
compile(code,'remote_engineering_metadata842.py','exec');packet.mkdir();(packet/'remote_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status();client.close();(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n');assert rc==0,error.decode()
value=json.loads(data);print(json.dumps(value,indent=2)[:7000])
