"""Run the explicit CPU witness using staged sources, without modifying execution code."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import shlex
import paramiko

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/global_task_role_cpu_witness')
assert not packet.exists();packet.mkdir()
stage='/tmp/trifusion_global_task_role_cpu_20261004'
sources={'trifusion/global_task_role_heads.py':repo/'modeling/trifusion/global_task_role_heads.py',
         'check_global_task_role_heads.py':repo/'tools/check_global_task_role_heads.py'}
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
_,out,err=client.exec_command(shlex.join(['/usr/bin/python3','-B','-c',f"from pathlib import Path;p=Path({stage!r});assert not p.exists();(p/'trifusion').mkdir(parents=True)"]))
error=err.read();assert out.channel.recv_exit_status()==0,error.decode()
sftp=client.open_sftp()
for name,path in sources.items():sftp.put(str(path),stage+'/'+name)
sftp.close()
sha={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()}
code=f'''from pathlib import Path
import hashlib,runpy,sys
stage=Path({stage!r})
for name,digest in {sha!r}.items():assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==digest
sys.path.insert(0,'/data/gaob/Re-ID/Trifusion/modeling')
import trifusion
trifusion.__path__.insert(0,str(stage/'trifusion'))
sys.argv=[str(stage/'check_global_task_role_heads.py'),'--output',str(stage/'WITNESS.json')]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
command='CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 '+shlex.join(['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-'])
stdin,out,err=client.exec_command(command)
stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(60)
data,error=out.read(),err.read();status=out.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'remote_cpu_source.py').write_text(code,encoding='utf-8')
(packet/'EXIT.json').write_text(json.dumps({'exit_code':status,'at':datetime.now().astimezone().isoformat(),'staged_source_sha256':sha,'physical_gpu_use':[]})+'\n',encoding='utf-8')
client.close();assert status==0,error.decode()
record=json.loads(data);assert record['status']=='CPU_HEAD_OWNERSHIP_WITNESS_PASS'
print(json.dumps(record))
