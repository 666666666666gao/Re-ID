"""Launch only the two never-started100 diagnoses after completed text sync."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import paramiko

REPO=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
BASE=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
PACKET=BASE/'deployment_metric_pending_fixed_best_launch854'
ROOT='/data/gaob/Re-ID/Trifusion'
LAUNCH=ROOT+'/logs/deployment_metric_pending_fixed_best_launch_20261005_854'
OUTPUT=ROOT+'/results/deployment_metric_fixed_best_pending100_20261005_854'
DIRECTORY='refine-logs/deployment_metric_pending_fixed_best_v1'
proof=json.loads(Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002/four_copy854_2025_pending.json').read_bytes())
assert proof['status']=='FOUR_COPIES_VERIFIED_2025_MIRROR_IO_PENDING'
owned={DIRECTORY+'/'+n:hashlib.sha256((REPO/DIRECTORY/n).read_bytes()).hexdigest() for n in ('CONTINUE_RGBNT100.py','EXPERIMENT_PLAN.md')}
assert all(proof['servers'][0]['verified_files'][n]==d for n,d in owned.items())
ast.parse((REPO/DIRECTORY/'CONTINUE_RGBNT100.py').read_text())
failure=json.loads((BASE/'deployment_metric_fixed_best_failed854_intake/stdout.json').read_bytes())
frozen={n:v['sha256'] for n,v in failure['files'].items() if n.endswith(('campaign.json','EXIT.json','DIAGNOSIS.json','MSVR310_semantic.log'))}
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',ROOT+'/'+DIRECTORY+'/CONTINUE_RGBNT100.py']
supervisor=f'''from datetime import datetime
from pathlib import Path
import json,os,subprocess
launch=Path({LAUNCH!r})
with (launch/'console.log').open('x') as log:
 result=subprocess.run({command!r},cwd={ROOT!r},env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')
raise SystemExit(result.returncode)
'''
code=f'''from datetime import datetime
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path({ROOT!r});launch=Path({LAUNCH!r});output=Path({OUTPUT!r})
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=={proof['head']!r}
assert all(sha(root/n)==d for n,d in {owned!r}.items())
assert all(sha(root/n)==d for n,d in {frozen!r}.items())
seal_path=root/'refine-logs/deployment_metric_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(seal_path)=='a4de9faca7f502cb7a4fa01800e437919810d2cfa83bfcf0361047d831a42cb7'
seal=json.loads(seal_path.read_text())
assert len(seal['source_sha256'])==341 and len(seal['artifact_sha256'])==271
assert all(sha(root/n)==d for n,d in seal['source_sha256'].items())
assert all(sha(Path(n))==d for n,d in seal['artifact_sha256'].items())
assert not Path('/proc/444043/stat').exists()
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
assert shutil.disk_usage(root).free>2*1024**3
assert not launch.exists() and not output.exists()
launch.mkdir();(launch/'supervisor.py').write_text({supervisor!r})
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
record=dict(status='TWO_NEVER_STARTED100_FIXED_DIAGNOSES_LAUNCHED_ONCE',at=datetime.now().astimezone().isoformat(),
 pid=process.pid,start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),source_commit={proof['head']!r},
 launch_dir=str(launch),output_dir=str(output),command={command!r},planned_models=2,
 administrative_source_sha256={owned!r},original_failure_frozen_sha256={frozen!r},
 scientific_source_count=339,original_diagnosis_source_count=341,artifact_count=271,
 gpu_memory_only=devices,disk_free_bytes=shutil.disk_usage(root).free,
 boundary='Only first RGBNT100 semantic/native fixed diagnosis. Original201 complete/MSVRfailed unchanged. No active remote text sync, training, retry, new selection or power-temperature action.')
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
compile(supervisor,'<pending854-supervisor>','exec');compile(code,'<pending854-launch>','exec')
assert not PACKET.exists();PACKET.mkdir();(PACKET/'REMOTE_SOURCE.py').write_bytes(code.encode())
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(300)
data,error=stdout.read(),stderr.read();status=stdout.channel.recv_exit_status();client.close()
(PACKET/'stdout.json').write_bytes(data);(PACKET/'stderr.txt').write_bytes(error)
(PACKET/'EXIT.json').write_bytes((json.dumps(dict(exit_code=status,at=datetime.now().astimezone().isoformat()))+'\n').encode())
assert status==0,error.decode();print(data.decode())
