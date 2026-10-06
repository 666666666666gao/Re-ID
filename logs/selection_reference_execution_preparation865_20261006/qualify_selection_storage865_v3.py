from pathlib import Path
import json
import paramiko

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selection_storage865_v3')
assert not packet.exists();packet.mkdir()
source=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selection_storage865_v2/SOURCE.py').read_text()
source=source.replace(" assert str(folder/'best_map.pth') not in protected", " if name in targets:assert str(folder/'best_map.pth') not in protected")
compile(source,'qualification865_v3','exec');(packet/'SOURCE.py').write_text(source)
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(source);i.channel.shutdown_write();o.channel.settimeout(300)
data,error=o.read(),e.read();status=o.channel.recv_exit_status();c.close()
(packet/'QUALIFICATION.json').write_bytes(data);(packet/'STDERR.txt').write_bytes(error)
assert status==0,error.decode()
d=json.loads(data);print(json.dumps(dict(status=d['status'],free_bytes=d['free_bytes'],qualified=len(d['qualified']),bytes=sum(r['target']['bytes'] for r in d['qualified']))))
