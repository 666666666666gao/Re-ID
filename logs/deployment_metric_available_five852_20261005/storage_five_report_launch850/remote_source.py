from datetime import datetime
from pathlib import Path
import hashlib,json,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
launch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_storage_five_report_launch_20261005_850')
output=Path('/data/gaob/Re-ID/Trifusion/results/deployment_metric_storage_five_20261005_848')
report=root/'refine-logs/deployment_metric_role_v1/REPORT_STORAGE_FIVE_848.py'
assert hashlib.sha256(report.read_bytes()).hexdigest()=='545c115b3402337285c88e8870e961f7fba36eeae7055fba4c92d00e169755b2'
state=json.loads((root/'logs/deployment_metric_storage_continuation_20261005_848/campaign.json').read_text())
assert state['status']=='COMPLETE' and 'active_command' not in state
assert len(state['jobs'])==3 and all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
terminal=json.loads((root/'logs/deployment_metric_storage_launch_20261005_848/EXIT.json').read_text())
assert terminal['exit_code']==0
assert state['report_invocations']==0
assert not output.exists() and not launch.exists()
launch.mkdir()
(launch/'supervisor.py').write_text("from datetime import datetime\nfrom pathlib import Path\nimport json,os,subprocess\nlaunch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_storage_five_report_launch_20261005_850')\ncommand=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_role_v1/REPORT_STORAGE_FIVE_848.py']\nwith (launch/'report.log').open('x') as log:\n result=subprocess.run(command,cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(result.returncode)\n")
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
record=dict(status='STORAGE_FIVE_CPU_REPORT_LAUNCHED_ONCE',started_at=datetime.now().astimezone().isoformat(),pid=process.pid,
 start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),launch_dir=str(launch),output_dir=str(output),
 report_source_sha256='545c115b3402337285c88e8870e961f7fba36eeae7055fba4c92d00e169755b2',continuation_completed_at=state['completed_at'],
 cuda_visible_devices='',boundary='Saved-array CPU report only; original failed campaigns and report counters unchanged; 12 available comparisons, three explicit missing pairs. No model inference or training.')
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
