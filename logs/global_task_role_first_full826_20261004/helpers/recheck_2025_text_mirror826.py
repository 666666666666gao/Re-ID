"""One read-only check of the previously failing text directory after2.5hours."""
from datetime import datetime
import json
from pathlib import Path
import paramiko

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/mirror2025_readonly826')
assert not packet.exists()
code='''
from datetime import datetime
from pathlib import Path
import hashlib,json,os
root=Path('/data2/gb/Re-ID/Trifusion')
directory=os.stat(root)
doc=root/'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
data=doc.read_bytes()
print(json.dumps(dict(status='TEXT_DIRECTORY_AND_HANDOFF_READABLE',at=datetime.now().astimezone().isoformat(),root=str(root),directory_inode=directory.st_ino,
 doc_bytes=len(data),doc_sha256=hashlib.sha256(data).hexdigest(),gpu_accessed=False,writes=0)))
'''
packet.mkdir()
(packet/'remote_readonly_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2025,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
out.channel.settimeout(60)
data,error=out.read(),err.read()
status=out.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data)
(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status,at=datetime.now().astimezone().isoformat(),gpu_accessed=False,writes=0))+'\n',encoding='utf-8')
client.close()
print(json.dumps(dict(exit_code=status,packet=str(packet),stdout=data.decode(),stderr=error.decode())))
