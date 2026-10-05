"""Launch one administrative evaluation/report completion; never relaunch training."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import paramiko

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet = base / 'capacity_first_eval_completion_launch857'
coordinator = Path('C:/Users/gb/.codex_tmp/capacity_first_eval_completion857_ADMIN.py').read_bytes()
retirement = json.loads((base / 'capacity_first_eval_closed_metric_retirement857/stdout.json').read_bytes())
failure_hashes = retirement['original_failure_sha256']
supervisor = r'''from datetime import datetime
import json,os
from pathlib import Path
import subprocess,sys
root=Path('/data/gaob/Re-ID/Trifusion')
launch=Path(__file__).resolve().parent
command=[sys.executable,'-B',str(root/'logs/semantic_capacity_first_eval_completion_20261005_857/ADMIN_COMPLETE857.py')]
with (launch/'console.log').open('x') as output:
 code=subprocess.run(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=output,stderr=subprocess.STDOUT).returncode
(launch/'EXIT.json').write_bytes((json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\n').encode())
raise SystemExit(code)
'''
code = rf'''from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/semantic_capacity_first_eval_completion_20261005_857'
launch=root/'logs/semantic_capacity_first_eval_launch_20261005_857'
original=root/'logs/semantic_capacity_control_v1_20261005_856'
assert not campaign.exists() and not launch.exists()
assert not (root/'results/semantic_capacity_control_v1_complete_20261005_856').exists()
assert not Path('/proc/642951').exists() and not Path('/proc/642952').exists()
assert shutil.disk_usage(root).free>=2*1024**3
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
 return h.hexdigest()
failure={failure_hashes!r}
assert all(sha(Path(n))==d for n,d in failure.items())
state=json.loads((original/'campaign.json').read_text())
full=next(r for r in state['jobs'] if (r['dataset'],r['phase'])==('RGBNT100','full'))
assert len(full['steps'])==1 and full['steps'][0]['mode']=='train' and full['steps'][0]['exit_code']==0
assert state['report_invocations']==0
run=root/'trained-model/semantic_capacity_control_v1_20261005_856_full_native_RGBNT100'
assert not (run/'official_metrics.json').exists() and not (original/'RGBNT100_evaluate.log').exists()
training=json.loads((run/'training.json').read_text())
assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and len(training['history'])==50
campaign.mkdir();launch.mkdir()
entry=campaign/'ADMIN_COMPLETE857.py';entry.write_bytes({coordinator!r})
assert sha(entry)=={hashlib.sha256(coordinator).hexdigest()!r}
plan=dict(status='FIRST_EVALUATION_AND_FIRST_REPORT_ONLY',registered_at=datetime.now().astimezone().isoformat(),
 original_failure_sha256=failure,original_parent_exit_code=1,original_campaign=str(original),
 training_sha256=sha(run/'training.json'),best_sha256=sha(run/'best_map.pth'),
 coordinator_sha256=sha(entry),science_source_count=345,raw_control_artifacts=187,
 new_training_invocations=0,first_strict100_previously_invoked=False,
 report_previously_invoked=False,physical_gpus=[0,1],disk_reserve_bytes=2*1024**3,
 boundary='Finish existing registered capacity study without rerunning training, changing weights/selection/recipe/source, or modifying original failure state/EXIT/log.100 M0 probe is kept until strict acceptance. No power/temperature action. New administrative state has inherited original training provenance.')
(campaign/'PLAN.json').write_bytes((json.dumps(plan,indent=2)+'\n').encode())
script=launch/'SUPERVISOR.py';script.write_bytes({supervisor.encode()!r})
with (launch/'supervisor.log').open('x') as output:
 process=subprocess.Popen(['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(script)],cwd=root,
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
ticks=Path('/proc/'+str(process.pid)+'/stat').read_text().split()[21]
value=dict(status='ADMINISTRATIVE_FIRST_STRICT_AND_REPORT_LAUNCHED',at=datetime.now().astimezone().isoformat(),
 pid=process.pid,start_ticks=ticks,campaign=str(campaign),launch=str(launch),
 coordinator_sha256=sha(entry),plan_sha256=sha(campaign/'PLAN.json'),
 original_failure_sha256=failure,new_training_invocations=0,disk_free_bytes=shutil.disk_usage(root).free)
(launch/'LAUNCH.json').write_bytes((json.dumps(value,indent=2)+'\n').encode())
print(json.dumps(value))
'''
compile(code, 'launch_capacity_first_eval_completion857.py', 'exec')
compile(supervisor, 'capacity_first_eval_supervisor857.py', 'exec')
assert not packet.exists()
packet.mkdir()
(packet / 'REMOTE_SOURCE.py').write_bytes(code.encode())
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(60)
data, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
client.close()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_bytes((json.dumps(dict(exit_code=status, at=datetime.now().astimezone().isoformat())) + '\n').encode())
assert status == 0, error.decode()
print(data.decode())
