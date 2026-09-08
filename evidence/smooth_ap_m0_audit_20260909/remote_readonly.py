"""Private transport, reads or CPU arithmetic only; never emits helper contents."""
import contextlib, io, json, os, shlex, sys, time
from pathlib import Path
OUT = Path(__file__).resolve().parent
payload = Path(sys.argv[1]).resolve()
assert payload.parent == OUT
helper = Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py')
private = {'__name__': 'audit_private_transport'}
with contextlib.redirect_stdout(io.TextIOWrapper(io.BytesIO(),encoding='utf-8')), contextlib.redirect_stderr(io.TextIOWrapper(io.BytesIO(),encoding='utf-8')):
    exec(compile(helper.read_bytes(), '<private-ssh-helper>', 'exec'), private)
c = private['c']
command = 'env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/tri_reid/bin/python -B -u -'
start = time.time()
stdin, stdout, stderr = c.exec_command(command, timeout=1200)
stdin.write(payload.read_text(encoding='utf-8'));stdin.flush();stdin.channel.shutdown_write()
out = stdout.read();err = stderr.read();code = stdout.channel.recv_exit_status()
c.close()
(OUT/(payload.stem+'.stdout')).write_bytes(out)
(OUT/(payload.stem+'.stderr')).write_bytes(err)
meta = {'command': command, 'script': str(payload), 'exit_code':code, 'elapsed_seconds': time.time()-start, 'helper_contents_and_credentials': 'never captured or published'}
(OUT/(payload.stem+'.execution.json')).write_text(json.dumps(meta, indent=2),encoding='utf-8')
print(json.dumps(meta,indent=2))
if code:
    print(err.decode('utf-8'))
    raise SystemExit(code)
