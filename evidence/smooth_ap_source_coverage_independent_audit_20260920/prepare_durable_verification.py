from pathlib import Path
import json,hashlib
out=Path(__file__).parent
src=out/'snapshots/prior_audit/independent_remote_cpu_verify.py'
data=src.read_bytes()
compile(data,str(src),'exec')
(out/'independent_remote_cpu_verify.py').write_bytes(data)
remote='/root/trifusion-storage/artifacts/smooth_ap_source_coverage_independent_audit_20260920'
wrapper='''from pathlib import Path
import subprocess,json,datetime,os,hashlib
root=Path(__file__).parent
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
start=stamp()
with (root/'independent_verify.jsonl').open('xb') as stdout,(root/'independent_verify.stderr').open('xb') as stderr:
    child=subprocess.Popen(['/root/miniconda3/envs/tri_reid/bin/python','-B',str(root/'independent_remote_cpu_verify.py')],stdout=stdout,stderr=stderr,env=env,cwd='/root',start_new_session=True)
    launch=dict(wrapper_pid=os.getpid(),child_pid=child.pid,started_at=start,script_sha256=hashlib.sha256((root/'independent_remote_cpu_verify.py').read_bytes()).hexdigest(),cpu_only=True)
    (root/'launch.json').write_text(json.dumps(launch,indent=2))
    rc=child.wait()
    (root/'terminal.json').write_text(json.dumps(dict(**launch,ended_at=stamp(),exit_code=rc,stdout_sha256=hashlib.sha256((root/'independent_verify.jsonl').read_bytes()).hexdigest(),stderr_sha256=hashlib.sha256((root/'independent_verify.stderr').read_bytes()).hexdigest()),indent=2))
'''
(out/'remote_wrapper.py').write_text(wrapper,encoding='utf-8')
payload='''from pathlib import Path
import subprocess,json,hashlib,datetime
root=Path(REMOTE)
assert not root.exists(), 'Audit directory already exists; inspect its launch and terminal receipts; do not relaunch.'
root.mkdir()
script=SCRIPT
wrapper=WRAPPER
(root/'independent_remote_cpu_verify.py').write_text(script,encoding='utf-8')
(root/'wrapper.py').write_text(wrapper,encoding='utf-8')
assert hashlib.sha256((root/'independent_remote_cpu_verify.py').read_bytes()).hexdigest()==SHA
with (root/'wrapper.stdout').open('xb') as stdout,(root/'wrapper.stderr').open('xb') as stderr:
    proc=subprocess.Popen(['/root/miniconda3/envs/tri_reid/bin/python','-B',str(root/'wrapper.py')],stdout=stdout,stderr=stderr,cwd='/root',start_new_session=True)
print(json.dumps(dict(status='LAUNCHED_CPU_INDEPENDENT_AUDIT',pid=proc.pid,root=str(root),script_sha256=SHA,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True)
'''.replace('REMOTE',repr(remote)).replace('SCRIPT',repr(data.decode('utf-8'))).replace('WRAPPER',repr(wrapper)).replace('SHA',repr(hashlib.sha256(data).hexdigest()))
(out/'launch_remote_verification.py').write_text(payload,encoding='utf-8')
print(json.dumps(dict(script_sha256=hashlib.sha256(data).hexdigest(),remote=remote)))
