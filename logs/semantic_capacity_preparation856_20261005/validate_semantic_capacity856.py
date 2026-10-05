"""Stage only six new files, inspect existing dependencies, run CPU witness."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/semantic_capacity_validation856')
names=['modeling/trifusion/semantic_capacity_evidence.py','tools/run_semantic_capacity.py',
       'tools/check_semantic_capacity.py','tools/queue_semantic_capacity.py','tools/report_semantic_capacity.py']
# Five files: reader and roles share one module; no old implementation edited.
assert len(names)==5 and not packet.exists();packet.mkdir()
digests={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in names}
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
sftp=client.open_sftp()
listing={n:{p.filename for p in sftp.listdir_attr('/data/gaob/Re-ID/Trifusion/'+str(Path(n).parent).replace('\\','/'))} for n in names}
assert all(Path(n).name not in listing[n] for n in names)
for name in names:
    with sftp.open('/data/gaob/Re-ID/Trifusion/'+name,'wb') as f:f.write((repo/name).read_bytes())
sftp.close()
code=f'''from pathlib import Path
from datetime import datetime
import ast,hashlib,json,subprocess,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
old=json.loads((root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(old)==339
assert all(sha(root/n)==d for n,d in old.items())
controls=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(controls['artifact_sha256'])==187 and len(controls['rows'])==9
assert all(sha(Path(n))==d for n,d in controls['artifact_sha256'].items())
for n,d in {digests!r}.items():
 assert sha(root/n)==d;ast.parse((root/n).read_text())
directory=root/'logs/semantic_capacity_validation_20261005_856'
assert not directory.exists();directory.mkdir()
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/check_semantic_capacity.py'),'--output',str(directory/'CPU_COMPONENT.json')]
with (directory/'component.log').open('x') as log:
 result=subprocess.run(command,cwd=root,env=dict(__import__('os').environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
value=dict(status='CPU_COMPONENT_PASS' if result.returncode==0 else 'CPU_COMPONENT_FAILED',at=datetime.now().astimezone().isoformat(),
 old_sources=339,controls=187,new_source_sha256={digests!r},component_exit_code=result.returncode,disk_free_bytes=shutil.disk_usage(root).free)
(directory/'VALIDATION.json').write_text(json.dumps(value,indent=2)+'\\n');print(json.dumps(value))
print((directory/'component.log').read_text())
assert result.returncode==0
'''
compile(code,'validate_remote.py','exec');(packet/'REMOTE_SOURCE.py').write_bytes(code.encode())
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(600)
data,error=stdout.read(),stderr.read();status=stdout.channel.recv_exit_status()
(packet/'stdout.txt').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_bytes((json.dumps(dict(exit_code=status,at=datetime.now().astimezone().isoformat()))+'\n').encode())
sftp=client.open_sftp()
for name in ('CPU_COMPONENT.json','VALIDATION.json','component.log'):
    target=packet/name
    with sftp.open('/data/gaob/Re-ID/Trifusion/logs/semantic_capacity_validation_20261005_856/'+name,'rb') as f:target.write_bytes(f.read())
sftp.close();client.close()
assert status==0,error.decode()+data.decode()
print(data.decode())
