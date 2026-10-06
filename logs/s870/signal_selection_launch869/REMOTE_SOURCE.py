expected_head='c1a70890726843897d619a62e423119adce7a45e'
supervisor="from pathlib import Path\nfrom datetime import datetime\nimport json,os,subprocess\nroot=Path('/data/gaob/Re-ID/Trifusion')\nlaunch=root/'logs/signal_selection_reference_launch_20261006_869'\ncommand=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/resume_signal_selection_disk.py'),'--campaign',str(root/'logs/signal_selection_reference_v1_20261006_869'),'--report-dir',str(root/'results/signal_selection_reference_complete_20261006_869')]\nenv=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')\nwith (launch/'stdout.txt').open('x') as output,(launch/'stderr.txt').open('x') as error:\n child=subprocess.Popen(command,cwd=root,env=env,stdout=output,stderr=error)\n row=dict(pid=child.pid,start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),started_at=datetime.now().astimezone().isoformat(),command=command,physical_gpus=[0,1])\n (launch/'CHILD.json').write_text(json.dumps(row,indent=2)+'\\n')\n code=child.wait()\n(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()),indent=2)+'\\n')\nraise SystemExit(code)\n"
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess,os
root=Path('/data/gaob/Re-ID/Trifusion')
launch=root/'logs/signal_selection_reference_launch_20261006_869'
assert not launch.exists() and not (root/'logs/signal_selection_reference_v1_20261006_869').exists()
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==expected_head
active=[]
for directory in Path('/proc').iterdir():
 if directory.name.isdigit() and (directory/'cmdline').is_file():
  command=(directory/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if '/data/gaob/Re-ID/Trifusion/tools/' in command:active.append(dict(pid=directory.name,command=command))
assert not active,active
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
scope=root/'refine-logs/signal_selection_reference_v1/SOURCE_SCOPE.json'
sources=json.loads(scope.read_text())['source_sha256']
assert len(sources)==374 and all(sha(root/name)==digest for name,digest in sources.items())
overlay=root/'refine-logs/signal_selection_disk_continuation_v1/SOURCE_SCOPE.json'
extra=json.loads(overlay.read_text())['source_sha256'];assert len(extra)==4 and all(sha(root/n)==d for n,d in extra.items())
controls=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
protected=json.loads(controls.read_text())['artifact_sha256']
assert len(protected)==187 and all(sha(name)==digest for name,digest in protected.items())
gpu=subprocess.check_output(['nvidia-smi','-i','0,1','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in gpu.splitlines())
free=shutil.disk_usage(root).free
assert free>=3634392494,(free,3634392494)
assert json.loads((root/'logs/signal_selection_reference_launch_20261006_868/EXIT.json').read_text())['exit_code']==1
probe=root/'trained-model/signal_selection_reference_v1_20261006_868_m0_masked_RGBNT100/m0_reload_probe.pth'
assert sha(probe)=='615c754c4a04abdc2e937fc151a6fe2c979213b526a8bd59a88357a445b033cf'
runtime=Path('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')
assert runtime.is_file() and os.access(runtime,os.X_OK)
protocols={dataset:str(root/f'logs/training_feature_scale_protocols_20261002/{dataset}.json') for dataset in ('RGBNT201','MSVR310','RGBNT100')}
public=root/'pertrained-model/ViT-B-16.pt'
settings=dict(at=datetime.now().astimezone().isoformat(),status='PUBLISHED_SOURCE_CAPACITY_AND_PROTECTED_INPUTS_VERIFIED',head=expected_head,free_bytes=free,storage_required_bytes=3634392494,existing_masked_probe_bytes=probe.stat().st_size,gpu_memory_only=gpu,source_count=len(sources),source_scope_sha256=sha(scope),protected_count=len(protected),protected_seal_sha256=sha(controls),public_clip_sha256=sha(public),protocol_sha256={d:sha(p) for d,p in protocols.items()},runtime=str(runtime),boundary='Warm unchanged tri_reid, sevencompletedreused/twomissing source references; eightacceptedM0reused/oneallpatchM0new; sourcepublishedbeforeoneinvocation; budgetincludesfuturebests/probe/savepeak/distancearrays and2GiB, checkedagainbycontroller; no power/temp/25/otherproject action; GPU0/1 one NN.')
launch.mkdir(parents=True)
(launch/'QUALIFICATION.json').write_text(json.dumps(settings,indent=2)+'\n')
(launch/'supervisor.py').write_text(supervisor)
with (launch/'supervisor.stdout.txt').open('x') as output,(launch/'supervisor.stderr.txt').open('x') as error:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdout=output,stderr=error,start_new_session=True)
result=dict(status='SUPERVISOR_STARTED_NO_M0_OR_FORMAL_ACCEPTANCE',pid=process.pid,start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().split()[21]),started_at=datetime.now().astimezone().isoformat(),launch_dir=str(launch),qualification=settings)
(launch/'LAUNCH.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
