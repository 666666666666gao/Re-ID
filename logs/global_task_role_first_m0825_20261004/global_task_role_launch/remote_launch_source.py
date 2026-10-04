
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='f5ce44d69da6e7d2e33cfd48e5341fa57ba34f7c'
scope=root/'refine-logs/global_task_role_v1/SOURCE_SCOPE.json'
assert hashlib.sha256(scope.read_bytes()).hexdigest()=='5ac2b38e74cb837d902205482ecfdc32686d3698739300ce94c68b68ab70eb34'
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
supervisor="from pathlib import Path\nfrom datetime import datetime\nimport json,os,subprocess\nlaunch=Path("+repr(str(launch))+")\ncommand="+repr(command)+"\nwith (launch/'console.log').open('x') as log:\n result=subprocess.run(command,cwd="+repr(str(root))+",env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(result.returncode)\n"
compile(supervisor,'supervisor.py','exec')
launch.mkdir()
(launch/'supervisor.py').write_text(supervisor)
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,
 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
record=dict(status='LAUNCHED_ONCE',at=datetime.now().astimezone().isoformat(),pid=process.pid,start_ticks=ticks,
 launch_dir=str(launch),campaign=str(campaign),report_dir=str(output),command=command,source_commit='f5ce44d69da6e7d2e33cfd48e5341fa57ba34f7c',
 scope_sha256='5ac2b38e74cb837d902205482ecfdc32686d3698739300ce94c68b68ab70eb34',source_files_verified=330,gpu_observation=devices,disk_free_bytes=shutil.disk_usage(root).free,
 boundary='Only26 physicalGPU0/1,one paired queue,own8M0 then fresh50 then first strict eval per endpoint. No power/temp actions,no N2/N3,no new seed/LR/gain/batch,no retired M0 replay. Launch is not model M0 or performance evidence.')
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
