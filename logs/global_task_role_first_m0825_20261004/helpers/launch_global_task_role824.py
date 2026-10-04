"""Launch the sealed six-endpoint queue once on2026 physicalGPU0/1."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private = Path('C:/Users/gb/.codex_tmp')
base = private / 'independent_evidence_draft'
publication = json.loads((private / 'foundation_recipe_v1_20261002/four_copy824_2025_pending.json').read_bytes())
assert publication['status'] == 'FOUR_COPIES_VERIFIED_2025_MIRROR_IO_PENDING'
assert len(publication['servers']) == 1 and publication['servers'][0]['port'] == 2026
assert hashlib.sha256((base / 'publication822_mirror_io_check/CHECK.json').read_bytes()).hexdigest() == publication['pending_mirror']['failure_receipt_sha256']
for name in ('global_task_role_source_seal', 'global_task_role_configuration', 'global_task_role_cpu_witness'):
    assert json.loads((base / name / 'EXIT.json').read_bytes())['exit_code'] == 0
packet = base / 'global_task_role_launch'
assert not packet.exists()
scopepath = 'refine-logs/global_task_role_v1/SOURCE_SCOPE.json'
scope_sha = hashlib.sha256((repo / scopepath).read_bytes()).hexdigest()
code = f'''
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=={publication['head']!r}
scope=root/{scopepath!r}
assert hashlib.sha256(scope.read_bytes()).hexdigest()=={scope_sha!r}
bound=json.loads(scope.read_text())
assert len(bound['source_sha256'])==330
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in bound['source_sha256'].items())
for name,field in [('refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json','control_seal_sha256'),
 ('refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json','historical_control_seal_sha256')]:
 seal=root/name
 assert hashlib.sha256(seal.read_bytes()).hexdigest()==bound[field]
 formal=json.loads(seal.read_text())
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in formal['artifact_sha256'].items())
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
assert shutil.disk_usage(root).free>=6*2*384*1024**2+2*1024**3
campaign=root/'logs/global_task_role_v1_20261004_824'
output=root/'results/global_task_role_v1_complete_20261004_824'
launch=root/'logs/global_task_role_launch_20261004_824'
assert not campaign.exists() and not output.exists() and not launch.exists()
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/queue_global_task_role.py'),
 '--campaign',str(campaign),'--report-dir',str(output)]
supervisor="from pathlib import Path\\nfrom datetime import datetime\\nimport json,os,subprocess\\nlaunch=Path("+repr(str(launch))+")\\ncommand="+repr(command)+"\\nwith (launch/'console.log').open('x') as log:\\n result=subprocess.run(command,cwd="+repr(str(root))+",env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)\\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\\\n')\\nraise SystemExit(result.returncode)\\n"
compile(supervisor,'supervisor.py','exec')
launch.mkdir()
(launch/'supervisor.py').write_text(supervisor)
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,
 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
record=dict(status='LAUNCHED_ONCE',at=datetime.now().astimezone().isoformat(),pid=process.pid,start_ticks=ticks,
 launch_dir=str(launch),campaign=str(campaign),report_dir=str(output),command=command,source_commit={publication['head']!r},
 scope_sha256={scope_sha!r},source_files_verified=330,gpu_observation=devices,disk_free_bytes=shutil.disk_usage(root).free,
 boundary='Only26 physicalGPU0/1,one paired queue,own8M0 then fresh50 then first strict eval per endpoint. No power/temp actions,no N2/N3,no new seed/LR/gain/batch,no retired M0 replay. Launch is not model M0 or performance evidence.')
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
stdout.channel.settimeout(180)
data,error=stdout.read(),stderr.read()
exit_code=stdout.channel.recv_exit_status()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_text(json.dumps(dict(exit_code=exit_code,at=datetime.now().astimezone().isoformat()))+'\n',encoding='utf-8')
client.close()
assert exit_code==0,error.decode()
print(json.dumps(json.loads(data)))
