"""Read candidate historical nonprimary seed weights; preserve group winners."""
from pathlib import Path
from datetime import datetime
import json,re
import paramiko
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
inventory=json.loads((base/'deployment_metric_weight_inventory841/stdout.json').read_bytes())
paths=[r['path'] for r in inventory['weights'] if re.search('/official_extra_seed(4[3-9]|5[0-9])_',r['path']) and r['path'].endswith('/roles_epoch20.pth')]
assert len(paths)==25
packet=base/'old_seed_weight_receipts842';assert not packet.exists()
code='''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');rows=[]
for name in PATHS:
 p=Path(name);assert p.resolve().is_relative_to(root/'trained-model') and p.name=='roles_epoch20.pth'
 files={n:dict(text=(p.parent/n).read_bytes().decode('utf-8'),sha256=hashlib.sha256((p.parent/n).read_bytes()).hexdigest()) for n in ('training.json','official_metrics.json')}
 rows.append(dict(path=name,bytes=p.stat().st_size,checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),files=files))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),rows=rows,disk_free_bytes=shutil.disk_usage(root).free,boundary='Read-only old20epoch nonprimaryseed weights/receipts,no eligibility or deletion claim,no model/GPU/power-temperature action.')))
'''.replace('PATHS',repr(paths))
compile(code,'remote_old_seed_receipts842.py','exec');packet.mkdir();(packet/'remote_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status();client.close()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n')
assert rc==0,error.decode();value=json.loads(data)
print(json.dumps(dict(at=value['at'],count=len(value['rows']),example={n:json.loads(r['text']) for n,r in value['rows'][0]['files'].items()}),ensure_ascii=False)[:4200])
