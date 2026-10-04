
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
scope=root/'refine-logs/role_input_detach_v1/SOURCE_SCOPE.json'
seal=root/'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='90203d4d24a217cf19b0d2fbf02783d47fefd3cd'
assert hashlib.sha256(scope.read_bytes()).hexdigest()=='4d3fb75a21e04d826acde5b2a2c84f7c302c3ce684fa28e9855edbcd58a6f1e6'
assert hashlib.sha256(seal.read_bytes()).hexdigest()=='faec05a5eea483570b136a1c087e787c336f5fee2cf9d416b05c879d0bcee724'
sources=json.loads(scope.read_text())['source_sha256']
assert len(sources)==322
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in sources.items())
controls=json.loads(seal.read_text())
assert len(controls['rows'])==9 and len(controls['artifact_sha256'])==61
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in controls['artifact_sha256'].items())
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
campaign=Path('/data/gaob/Re-ID/Trifusion/logs/role_input_detach_v1_20261004_813');report=Path('/data/gaob/Re-ID/Trifusion/results/role_input_detach_v1_complete_20261004_813');launch=Path('/data/gaob/Re-ID/Trifusion/logs/role_input_detach_launch_20261004_813')
assert not campaign.exists() and not report.exists() and not launch.exists()
assert shutil.disk_usage(root).free >= 6*2*384*1024**2+2*1024**3
assert Path('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python').is_file()
launch.mkdir()
supervisor="from pathlib import Path\nfrom datetime import datetime\nimport json,os,subprocess\nlaunch=Path('/data/gaob/Re-ID/Trifusion/logs/role_input_detach_launch_20261004_813')\ncommand=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/tools/queue_role_input_detach.py', '--campaign', '/data/gaob/Re-ID/Trifusion/logs/role_input_detach_v1_20261004_813', '--report-dir', '/data/gaob/Re-ID/Trifusion/results/role_input_detach_v1_complete_20261004_813']\nwith (launch/'console.log').open('x') as log:\n    result=subprocess.run(command,cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(result.returncode)\n"
compile(supervisor,'supervisor.py','exec')
(launch/'supervisor.py').write_text(supervisor)
with (launch/'supervisor.log').open('x') as log:
    process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
record={'status':'LAUNCHED_ONCE','at':datetime.now().astimezone().isoformat(),'pid':process.pid,'start_ticks':ticks,
 'launch_dir':str(launch),'campaign_dir':str(campaign),'report_dir':str(report),'command':['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/tools/queue_role_input_detach.py', '--campaign', '/data/gaob/Re-ID/Trifusion/logs/role_input_detach_v1_20261004_813', '--report-dir', '/data/gaob/Re-ID/Trifusion/results/role_input_detach_v1_complete_20261004_813'],
 'source_scope_sha256':'4d3fb75a21e04d826acde5b2a2c84f7c302c3ce684fa28e9855edbcd58a6f1e6','control_seal_sha256':'faec05a5eea483570b136a1c087e787c336f5fee2cf9d416b05c879d0bcee724','source_commit':'90203d4d24a217cf19b0d2fbf02783d47fefd3cd',
 'source_count':len(sources),'unchanged_control_artifact_count':len(controls['artifact_sha256']),
 'gpu_observation':devices,'disk_free_bytes':shutil.disk_usage(root).free,
 'boundary':'Only26GPU0/1, one sequential two-card pair, six own8-update M0 then six fresh full50 and first strict reload. Existing warm environment unchanged. No power/temperature operations, parity or kernel repair. Historical nine controls unchanged, no old retired M0 verifier; scientific Goal active/unmet.'}
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
