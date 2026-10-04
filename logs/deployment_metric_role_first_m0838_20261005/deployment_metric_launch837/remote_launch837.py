from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='6b7a6f2afa13e57576d2981fc832008138b86909'
scope=root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json';assert hashlib.sha256(scope.read_bytes()).hexdigest()=='76987d2ca7d6dc7a9aeac34000d37ebe5e7d1d8ead6d52b1832f0b1557523a3d'
bound=json.loads(scope.read_text());assert len(bound['source_sha256'])==339
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in bound['source_sha256'].items())
control=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert hashlib.sha256(control.read_bytes()).hexdigest()==bound['control_seal_sha256']
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==d for n,d in json.loads(control.read_text())['artifact_sha256'].items())
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
assert shutil.disk_usage(root).free>=7*384*1024**2+2*1024**3
campaign,report,launch=map(Path,['/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_v1_20261005_837', '/data/gaob/Re-ID/Trifusion/results/deployment_metric_role_v1_complete_20261005_837', '/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_launch_20261005_837'])
assert not campaign.exists() and not report.exists() and not launch.exists()
launch.mkdir()
(launch/'supervisor.py').write_text("from pathlib import Path\nfrom datetime import datetime\nimport json,os,subprocess\nlaunch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_launch_20261005_837')\ncommand=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/tools/queue_deployment_metric_role.py', '--campaign', '/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_v1_20261005_837', '--report-dir', '/data/gaob/Re-ID/Trifusion/results/deployment_metric_role_v1_complete_20261005_837']\nwith (launch/'console.log').open('x') as log:\n result=subprocess.run(command,cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(result.returncode)\n")
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
record=dict(status='LAUNCHED_ONCE',at=datetime.now().astimezone().isoformat(),pid=process.pid,start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
 campaign=str(campaign),report_dir=str(report),launch_dir=str(launch),command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/tools/queue_deployment_metric_role.py', '--campaign', '/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_v1_20261005_837', '--report-dir', '/data/gaob/Re-ID/Trifusion/results/deployment_metric_role_v1_complete_20261005_837'],source_commit='6b7a6f2afa13e57576d2981fc832008138b86909',source_files_verified=339,
 scope_sha256='76987d2ca7d6dc7a9aeac34000d37ebe5e7d1d8ead6d52b1832f0b1557523a3d',gpu_memory_only=devices,disk_free_bytes=shutil.disk_usage(root).free,
 boundary='Only26GPU0/1,no power/temperature action. Own8M0 thenfresh50/firststrict perendpoint; onlyafterfull acceptance retire thatprobe withSHAjournal. Oldcontrols not replayed. Launch is notM0 or performance.')
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
