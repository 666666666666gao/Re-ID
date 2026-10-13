"""Launch the registered original finite two-control queue once."""
import json,subprocess
from datetime import datetime
from pathlib import Path

w=Path(__file__).resolve().parent
registration=json.loads((w/'VALUE_REGISTRATION_FINAL_951.json').read_bytes())
assert not (w/'VALUE_LAUNCH_REQUEST_951.json').exists()
request=dict(source_sha256=registration['source_sha256'],registration_commit=registration['commit'])
(w/'VALUE_LAUNCH_REQUEST_951.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),**request),indent=2)+'\n')
code='request='+repr(request)+'\n'+r'''
import hashlib,json,os,shutil,subprocess,time
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
admin=root/'logs/pretrained_value_launch951';campaign=root/'logs/pretrained_value_training951'
assert not admin.exists() and not campaign.exists()
assert all(sha(root/name)==digest for name,digest in request['source_sha256'].items())
assert shutil.disk_usage(root).free>=4026531840
gpu=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
used={int(line.split(',')[0]):int(line.split(',')[1]) for line in gpu.splitlines()}
assert used[0]<500 and used[1]<500
admin.mkdir()
supervisor=r"""
import json,os,subprocess
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');admin=root/'logs/pretrained_value_launch951'
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/queue_pretrained_value_delta.py'),'--campaign',str(root/'logs/pretrained_value_training951')]
with (admin/'controller.log').open('x') as log:
    child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    receipt=dict(supervisor_pid=os.getpid(),controller_pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    (admin/'RUNNING.json').write_text(json.dumps(receipt,indent=2)+'\n')
    status=child.wait()
receipt.update(exit_code=status,completed_at=datetime.now().astimezone().isoformat())
(admin/'EXIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
"""
(admin/'supervisor.py').write_text(supervisor)
(admin/'SOURCE_SEAL.json').write_text(json.dumps(request,indent=2)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
with (admin/'supervisor.log').open('x') as log:
    parent=subprocess.Popen(['/usr/bin/python3',str(admin/'supervisor.py')],cwd=root,env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(3)
receipt=json.loads((admin/'RUNNING.json').read_text())
proc=Path('/proc')/str(receipt['controller_pid'])
assert proc.is_dir() and (Path('/proc')/str(parent.pid)).is_dir()
receipt.update(actual_controller_cmdline=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
    actual_controller_cuda_visible_devices=next(x.split(b'=',1)[1].decode() for x in (proc/'environ').read_bytes().split(b'\0') if x.startswith(b'CUDA_VISIBLE_DEVICES=')),
    observed_at=datetime.now().astimezone().isoformat(),gpu_memory_mib=used,disk_free_bytes=shutil.disk_usage(root).free,
    registration_commit=request['registration_commit'],status='ORIGINAL_CONTROLLER_LIVE',
    conditions=['semantic_reference','uniform_reference'])
assert receipt['actual_controller_cuda_visible_devices']=='0,1'
(admin/'LAUNCH_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code,text=True,capture_output=True)
(w/'VALUE_LAUNCH_951.stdout').write_bytes(run.stdout.encode());(w/'VALUE_LAUNCH_951.stderr').write_bytes(run.stderr.encode())
(w/'VALUE_LAUNCH_EXIT_951.json').write_text(json.dumps(dict(exit_code=run.returncode,at=datetime.now().astimezone().isoformat()))+'\n')
assert run.returncode==0,run.stderr
receipt=json.loads(run.stdout);(w/'VALUE_LAUNCH_VERIFIED_951.json').write_text(json.dumps(receipt,indent=2)+'\n')
memo=f'\nTriFusion {receipt["observed_at"]} §951 actual finite2 value-path queue supervisor{receipt["supervisor_pid"]}/controller{receipt["controller_pid"]} live once/actualCUDA01,496sourceexact/Git{registration["commit"]}. Eachfreshprepare8M0then50firststrict; noactualM0/formalscoreyet. Only26physical01/rawauthor42/noT-P/envchange/seedtuning; fullall3SignalGoalACTIVE_UNMET. Next nearM0>=180sec, then nearfirstfullETA, no restart from observer timeout.\n'
for path in (Path('C:/Users/gb/memory/2026-10-14.md'),Path('C:/Users/gb/MEMORY.md')):
    with path.open('a',encoding='utf-8') as handle:handle.write(memo)
print(json.dumps(receipt))
