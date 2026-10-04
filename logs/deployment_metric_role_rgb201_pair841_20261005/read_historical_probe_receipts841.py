"""Read completed historical M0 receipts and exact probe hashes before retirement."""
from pathlib import Path
from datetime import datetime
import json
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
inventory=json.loads((base/'deployment_metric_weight_inventory841/stdout.json').read_bytes())
paths=[r['path'] for r in inventory['weights'] if r['path'].endswith('/m0_reload_probe.pth') and r['bytes']<100000000]
assert len(paths)==70
packet=base/'historical_probe_receipts841';assert not packet.exists()
code='''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');paths=PATHS
rows=[]
for name in paths:
 p=Path(name);assert p.resolve().is_relative_to(root/'trained-model') and p.name=='m0_reload_probe.pth'
 training=p.parent/'training.json';data=training.read_bytes();value=json.loads(data)
 rows.append(dict(path=name,bytes=p.stat().st_size,probe_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                  training_sha256=hashlib.sha256(data).hexdigest(),training_text=data.decode('utf-8')))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),rows=rows,disk_free_bytes=shutil.disk_usage(root).free,
 boundary='Historical small M0 probe/receipt read only;no deletion,model,GPU or power-temperature query. Current large live probe excluded.')))
'''.replace('PATHS',repr(paths))
compile(code,'remote_historical_probe_read841.py','exec')
packet.mkdir();(packet/'remote_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status();client.close()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n')
assert rc==0,error.decode()
value=json.loads(data)
print(json.dumps(dict(at=value['at'],rows=len(value['rows']),status_counts={s:sum(json.loads(r['training_text'])['status']==s for r in value['rows']) for s in sorted({json.loads(r['training_text'])['status'] for r in value['rows']})})))
