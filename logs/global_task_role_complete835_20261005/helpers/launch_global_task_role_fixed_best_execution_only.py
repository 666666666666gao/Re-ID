"""Launch the unchanged registered diagnosis after verifying the execution copies."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import paramiko

parser = argparse.ArgumentParser()
parser.add_argument('--publication-proof', type=Path, required=True)
args = parser.parse_args()
repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
publication = json.loads(args.publication_proof.read_bytes())
assert publication['status'] == 'FOUR_COPIES_VERIFIED_2025_MIRROR_IO_PENDING'
assert publication['pending_mirror']['port'] == 2025 and publication['pending_mirror']['status'] == 'INPUT_OUTPUT_ERROR'
assert hashlib.sha256((base / 'publication822_mirror_io_check/CHECK.json').read_bytes()).hexdigest() == publication['pending_mirror']['failure_receipt_sha256']
assert len(publication['servers']) == 1 and publication['servers'][0]['port'] == 2026
assert json.loads((base / 'global_task_role_complete_report/EXIT.json').read_bytes())['exit_code'] == 0
assert json.loads((base / 'global_task_role_fixed_best_seal/EXIT.json').read_bytes())['exit_code'] == 0
packet = base / 'global_task_role_fixed_best_launch'
assert not packet.exists()
sealpath = 'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
sealsha = hashlib.sha256((repo / sealpath).read_bytes()).hexdigest()
seal = json.loads((repo / sealpath).read_bytes())
assert seal['schema'] == 'trifusion-global-task-role-fixed-best-diagnosis-v1'
assert len(seal['source_sha256']) == 332 and len(seal['rows']) == 9
code = f'''
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
seal=root/{sealpath!r}
assert hashlib.sha256(seal.read_bytes()).hexdigest()=={sealsha!r}
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=={publication['head']!r}
bound=json.loads(seal.read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in bound['source_sha256'].items())
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in bound['artifact_sha256'].items())
campaign=root/'logs/global_task_role_v1_20261004_824'
state=json.loads((campaign/'campaign.json').read_text())
parent=json.loads((root/'logs/global_task_role_launch_20261004_824/EXIT.json').read_text())
assert parent['exit_code']==0 and state['status']=='COMPLETE'
assert state['report_invocations']==1 and state['report_exit_code']==0
assert len(state['jobs'])==12 and all(row['status']=='COMPLETE' and row['exit_code']==0 for row in state['jobs'])
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
output=root/'results/global_task_role_fixed_best_20261004_v1'
launch=root/'logs/global_task_role_fixed_best_launch_20261004_v1'
assert not output.exists() and not launch.exists()
assert shutil.disk_usage(root).free>2*1024**3
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/diagnose_global_task_role_best.py'),
 '--campaign',str(campaign),'--seal',str(seal),'--output-dir',str(output)]
supervisor="from pathlib import Path\\nfrom datetime import datetime\\nimport json,os,subprocess\\nlaunch=Path("+repr(str(launch))+")\\ncommand="+repr(command)+"\\nwith (launch/'console.log').open('x') as log:\\n result=subprocess.run(command,cwd="+repr(str(root))+",env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)\\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\\\n')\\nraise SystemExit(result.returncode)\\n"
compile(supervisor,'supervisor.py','exec')
launch.mkdir()
(launch/'supervisor.py').write_text(supervisor)
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
record={{'status':'LAUNCHED_ONCE','at':datetime.now().astimezone().isoformat(),'pid':process.pid,'start_ticks':ticks,
 'launch_dir':str(launch),'output_dir':str(output),'command':command,'seal_sha256':{sealsha!r},
 'source_commit':{publication['head']!r},'gpu_observation':devices,'disk_free_bytes':shutil.disk_usage(root).free,
 'boundary':'Registered six fixed best, sequential on26GPU0/1. No new training, checkpoint selection, test update, power/temperature action, original retired M0 verifier or report replay.'}}
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
compile(code, 'remote_launch_source.py', 'exec')
packet.mkdir()
(packet / 'remote_launch_source.py').write_text(code, encoding='utf-8')
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(60)
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_text(json.dumps({'exit_code': exit_code}) + '\n', encoding='utf-8')
client.close()
assert exit_code == 0, error.decode()
print(json.dumps(json.loads(data)))
