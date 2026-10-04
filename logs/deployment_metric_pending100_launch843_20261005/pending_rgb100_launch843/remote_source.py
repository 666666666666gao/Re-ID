from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion');launch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_pending100_launch_20261005_843');campaign=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_pending100_20261005_842')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='3f14d8410b78c4e8b38e5a8c81b068df9a536fbb'
assert hashlib.sha256((root/'refine-logs/deployment_metric_role_v1/PENDING_RGBNT100_QUEUE.py').read_bytes()).hexdigest()=='19ac57ad4172be5e60f0ff48a4b6acc41e6f5de4761701662a485075d3b88b94'
assert hashlib.sha256((root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_bytes()).hexdigest()=='76987d2ca7d6dc7a9aeac34000d37ebe5e7d1d8ead6d52b1832f0b1557523a3d'
scope=json.loads((root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_text());assert len(scope['source_sha256'])==339
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in scope['source_sha256'].items())
origin=root/'logs/deployment_metric_role_v1_20261005_837/campaign.json'
assert hashlib.sha256(origin.read_bytes()).hexdigest()=='88e2575a7d3693afc0e9e02d0ed3352e61e0eb02447263b8a5c4ff1612db62f9'
assert json.loads(origin.read_text())['status']=='FAILED'
assert not Path('/proc/3606472').exists()
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
assert shutil.disk_usage(root).free>=3*384*1024**2+2*1024**3
assert not campaign.exists() and not launch.exists();launch.mkdir()
(launch/'supervisor.py').write_text("from pathlib import Path\nfrom datetime import datetime\nimport json,os,subprocess\nlaunch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_pending100_launch_20261005_843')\ncommand=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_role_v1/PENDING_RGBNT100_QUEUE.py']\nwith (launch/'console.log').open('x') as log:\n result=subprocess.run(command,cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(result.returncode)\n")
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
record=dict(status='REGISTERED_PENDING_RGBNT100_PAIR_LAUNCHED_ONCE',at=datetime.now().astimezone().isoformat(),pid=process.pid,
 start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),campaign=str(campaign),launch_dir=str(launch),command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_role_v1/PENDING_RGBNT100_QUEUE.py'],
 source_commit='3f14d8410b78c4e8b38e5a8c81b068df9a536fbb',coordinator_sha256='19ac57ad4172be5e60f0ff48a4b6acc41e6f5de4761701662a485075d3b88b94',source_files_verified=339,origin_terminal_sha256='88e2575a7d3693afc0e9e02d0ed3352e61e0eb02447263b8a5c4ff1612db62f9',expected_endpoints=2,
 gpu_memory_only=devices,disk_free_bytes=shutil.disk_usage(root).free,boundary='Only originally never-started100 pair,same model/training339source,no originalfailure retry or rewrite. Own8M0/fresh50/firststrict;2/2not6/6. Only26GPU0/1,no power/temperature action. Launch notM0/performance.')
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
