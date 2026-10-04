
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
seal=root/'refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert hashlib.sha256(seal.read_bytes()).hexdigest()=='e997d2341ca86e875a6518162965182895a7778fbd84a99c7a3ffa70f82776e5'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='62184bc2d9fc4416b6e26da13d5eeb0a0dc010f2'
bound=json.loads(seal.read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in bound['source_sha256'].items())
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in bound['artifact_sha256'].items())
campaign=root/'logs/role_input_detach_v1_20261004_813'
state=json.loads((campaign/'campaign.json').read_text())
parent=json.loads((root/'logs/role_input_detach_launch_20261004_813/EXIT.json').read_text())
assert parent['exit_code']==0 and state['status']=='COMPLETE'
assert state['report_invocations']==1 and state['report_exit_code']==0
assert len(state['jobs'])==12 and all(row['status']=='COMPLETE' and row['exit_code']==0 for row in state['jobs'])
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
output=root/'results/role_input_detach_fixed_best_20261004_v1'
launch=root/'logs/role_input_detach_fixed_best_launch_20261004_v1'
assert not output.exists() and not launch.exists()
assert shutil.disk_usage(root).free>2*1024**3
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/diagnose_role_input_detach_best.py'),
 '--campaign',str(campaign),'--seal',str(seal),'--output-dir',str(output)]
supervisor="from pathlib import Path\nfrom datetime import datetime\nimport json,os,subprocess\nlaunch=Path("+repr(str(launch))+")\ncommand="+repr(command)+"\nwith (launch/'console.log').open('x') as log:\n result=subprocess.run(command,cwd="+repr(str(root))+",env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(result.returncode)\n"
compile(supervisor,'supervisor.py','exec')
launch.mkdir()
(launch/'supervisor.py').write_text(supervisor)
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
record={'status':'LAUNCHED_ONCE','at':datetime.now().astimezone().isoformat(),'pid':process.pid,'start_ticks':ticks,
 'launch_dir':str(launch),'output_dir':str(output),'command':command,'seal_sha256':'e997d2341ca86e875a6518162965182895a7778fbd84a99c7a3ffa70f82776e5',
 'source_commit':'62184bc2d9fc4416b6e26da13d5eeb0a0dc010f2','gpu_observation':devices,'disk_free_bytes':shutil.disk_usage(root).free,
 'boundary':'Registered six fixed best, sequential on26GPU0/1. No new training, checkpoint selection, test update, power/temperature action, original retired M0 verifier or report replay.'}
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
