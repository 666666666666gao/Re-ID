from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion');launch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_storage_launch_20261005_848');campaign=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_storage_continuation_20261005_848')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(root/'refine-logs/deployment_metric_role_v1/STORAGE_CONTINUATION_848.py')=='f4e347dc0b22a8d0b3ec226d01ea21a737ec6840c13711681afc64ffaf7a6aba'
assert sha(root/'refine-logs/deployment_metric_role_v1/STORAGE_CONTINUATION_PLAN_848.md')=='7c9550f6e1df08ee640788edfc786bd1171d2c248d8037924970c8730de1c14f'
scope=root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json'
assert sha(scope)=='76987d2ca7d6dc7a9aeac34000d37ebe5e7d1d8ead6d52b1832f0b1557523a3d'
sources=json.loads(scope.read_text())['source_sha256'];assert len(sources)==339
assert all(sha(root/n)==d for n,d in sources.items())
pending=root/'logs/deployment_metric_role_pending100_20261005_842/campaign.json'
assert sha(pending)=='4d5b9e00e2b3739a59c0ae411c89b28b4955701b5bb16780eaab84e54130223d'
old_exit=root/'logs/deployment_metric_pending100_launch_20261005_843/EXIT.json'
assert json.loads(old_exit.read_text())['exit_code']==1
assert not Path('/proc/3997841').exists() and not Path('/proc/3997842').exists()
assert not Path('/proc/4002738').exists()
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
assert shutil.disk_usage(root).free>=2*384*1024**2+2*1024**3
assert not launch.exists() and not campaign.exists();launch.mkdir()
(launch/'supervisor.py').write_text("from pathlib import Path\nfrom datetime import datetime\nimport json,os,subprocess\nlaunch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_storage_launch_20261005_848')\ncommand=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_role_v1/STORAGE_CONTINUATION_848.py']\nwith (launch/'console.log').open('x') as log:\n result=subprocess.run(command,cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(result.returncode)\n")
with (launch/'supervisor.log').open('x') as log:
 p=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(p.pid)+'/stat').read_text()
record=dict(status='STORAGE_CONTINUATION_LAUNCHED_ONCE',at=datetime.now().astimezone().isoformat(),pid=p.pid,
 start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),campaign=str(campaign),launch_dir=str(launch),command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_role_v1/STORAGE_CONTINUATION_848.py'],
 coordinator_sha256='f4e347dc0b22a8d0b3ec226d01ea21a737ec6840c13711681afc64ffaf7a6aba',plan_sha256='7c9550f6e1df08ee640788edfc786bd1171d2c248d8037924970c8730de1c14f',scientific_sources=339,
 original_pending_campaign_sha256=sha(pending),original_exit_sha256=sha(old_exit),
 disk_free_bytes=shutil.disk_usage(root).free,gpu_memory_only=devices,
 boundary='First existingsemantic evaluation;original native never-started,same scientificsources. Original training,parentEXIT1,campaign snapshot unchanged. No semantic retraining/MSVRnative retry/report846/power-temperature action.')
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
