
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
seal=root/'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert hashlib.sha256(seal.read_bytes()).hexdigest()=='faec05a5eea483570b136a1c087e787c336f5fee2cf9d416b05c879d0bcee724'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='f2480621ccc9f5ad7ee5de89a2a994433543b95c'
bound=json.loads(seal.read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in bound['source_sha256'].items())
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
output=root/'results/native_fixed_best_20261004_811'
launch=root/'logs/native_fixed_best_launch_20261004_811'
assert not output.exists() and not launch.exists()
assert shutil.disk_usage(root).free>2*1024**3
launch.mkdir()
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/diagnose_native_research_best.py'),
 '--campaign',str(root/'logs/native_research_v6_20261003_794'),'--seal',str(seal),'--output-dir',str(output)]
supervisor="from pathlib import Path\nfrom datetime import datetime\nimport json,os,subprocess\nlaunch=Path("+repr(str(launch))+")\ncommand="+repr(command)+"\nwith (launch/'console.log').open('x') as log:\n result=subprocess.run(command,cwd="+repr(str(root))+",env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(result.returncode)\n"
compile(supervisor,'supervisor.py','exec')
(launch/'supervisor.py').write_text(supervisor)
with (launch/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
record={'status':'LAUNCHED_ONCE','at':datetime.now().astimezone().isoformat(),'pid':process.pid,'start_ticks':ticks,
 'launch_dir':str(launch),'output_dir':str(output),'command':command,'seal_sha256':'faec05a5eea483570b136a1c087e787c336f5fee2cf9d416b05c879d0bcee724',
 'source_commit':'f2480621ccc9f5ad7ee5de89a2a994433543b95c','gpu_observation':devices,'disk_free_bytes':shutil.disk_usage(root).free,
 'boundary':'Only26GPU0/1, one sequential pair. No power/temperature operations or engineering repair. Read-only fixed weights; no new training, no old M0-dependent verifier.'}
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
