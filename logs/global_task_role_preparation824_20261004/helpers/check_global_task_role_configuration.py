from datetime import datetime
from pathlib import Path
import hashlib
import json
import shlex
import paramiko

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/global_task_role_configuration')
assert not packet.exists();packet.mkdir()
source=repo/'tools/run_global_task_role.py'
remote='/tmp/trifusion_global_task_role_cpu_20261004/run_global_task_role.py'
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
_,out,err=client.exec_command(shlex.join(['/usr/bin/python3','-B','-c',f"from pathlib import Path;assert not Path({remote!r}).exists()"]));error=err.read();assert out.channel.recv_exit_status()==0,error.decode()
sftp=client.open_sftp();sftp.put(str(source),remote);sftp.close()
code=f'''from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
source=Path({remote!r})
assert hashlib.sha256(source.read_bytes()).hexdigest()=={hashlib.sha256(source.read_bytes()).hexdigest()!r}
sys.path.insert(0,'/data/gaob/Re-ID/Trifusion')
sys.path.insert(0,'/data/gaob/Re-ID/Trifusion/modeling')
import trifusion
trifusion.__path__.insert(0,'/tmp/trifusion_global_task_role_cpu_20261004/trifusion')
spec=importlib.util.spec_from_file_location('tools.run_global_task_role',source)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
module.configure()
inner=module.previous.base.entry.entry
assert inner.AuthorHeadEvidence is module.GlobalTaskRoleHeads
assert inner.foundation.build_core is module.build_core
assert inner.foundation.loss_values is module.loss_values
assert inner.foundation.condition is module.condition
assert inner.SCHEMA==inner.foundation.SCHEMA==module.SCHEMA
args=argparse.Namespace(variant='semantic',baseline_sha256='cpu-placeholder',initialization=source)
condition=inner.foundation.condition(args)
assert condition['objective_gradient_policy']==module.POLICY
assert condition['visual_placement']=='single_process_original_full_batch_first6_cuda1_last6_and_heads_cuda0'
assert condition['author_training_objectives']==['shared_global','role_corrected_fused']
print(json.dumps({{'status':'CPU_CONFIGURATION_CHAIN_PASS','condition':condition,'schema':module.SCHEMA,'boundary':'Imports and explicit configuration wiring only; no model build, production M0, GPU or training.'}}))
'''
stdin,out,err=client.exec_command('CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 '+shlex.join(['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-']))
stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(60)
data,error=out.read(),err.read();status=out.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'remote_source.py').write_text(code,encoding='utf-8')
(packet/'EXIT.json').write_text(json.dumps({'exit_code':status,'at':datetime.now().astimezone().isoformat(),'entry_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})+'\n',encoding='utf-8')
client.close();assert status==0,error.decode()
print(json.dumps(json.loads(data)))
