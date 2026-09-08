import contextlib, hashlib, io, json, shlex, sys
from pathlib import Path
from inspect_inputs import record, OUT

script = Path(sys.argv[1]).resolve()
assert script.is_relative_to(OUT)
payload = record(script)
scope = {}
class Capture(io.StringIO):
    def reconfigure(self, **kwargs): pass
with contextlib.redirect_stdout(Capture()), contextlib.redirect_stderr(Capture()):
    exec(compile(Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'), '<private_connection_helper>', 'exec'), scope)
connection = scope['c']
stdin, stdout, stderr = connection.exec_command('PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /root/miniconda3/envs/tri_reid/bin/python -u -', timeout=300)
stdin.write(payload.decode('utf-8')); stdin.channel.shutdown_write()
data=stdout.read(); errors=stderr.read(); code=stdout.channel.recv_exit_status()
(OUT/(script.stem+'.stdout.json')).write_bytes(data)
(OUT/(script.stem+'.stderr.txt')).write_bytes(errors)
connection.close()
print(json.dumps({'script':str(script),'exit_code':code,'stdout_bytes':len(data),'stderr_bytes':len(errors),'output':str(OUT/(script.stem+'.stdout.json'))}))
if errors: print(errors.decode('utf-8',errors='replace')[:4000])
sys.exit(code)
