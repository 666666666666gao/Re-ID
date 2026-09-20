from pathlib import Path
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
