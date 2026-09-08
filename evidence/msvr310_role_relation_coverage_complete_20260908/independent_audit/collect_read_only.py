"""Run the auditor's stdin-only remote CPU script and preserve output locally."""
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
script = OUT / sys.argv[1]
tag = script.stem
recovery = Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py')
namespace = {'__name__': 'auditor_connection_recovery'}
class SilentBuffer(io.StringIO):
    def reconfigure(self, **kwargs):
        pass

with contextlib.redirect_stdout(SilentBuffer()), contextlib.redirect_stderr(SilentBuffer()):
    exec(compile(recovery.read_bytes(), str(recovery), 'exec'), namespace)
c = namespace['c']
command = "env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 /root/miniconda3/envs/tri_reid/bin/python -B -u -"
started = datetime.now(timezone.utc).isoformat()
stdin, stdout, stderr = c.exec_command(command)
stdin.write(script.read_text(encoding='utf-8'))
stdin.flush()
stdin.channel.shutdown_write()
with (OUT / (tag + '.stdout.jsonl')).open('wb') as stream:
    while data := stdout.read(1024 * 1024):
        stream.write(data)
errors = stderr.read()
(OUT / (tag + '.stderr.txt')).write_bytes(errors)
status = stdout.channel.recv_exit_status()
c.close()
receipt = dict(command=command, remote_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
               started_at=started, ended_at=datetime.now(timezone.utc).isoformat(), exit_code=status,
               stdout=dict(bytes=(OUT/(tag+'.stdout.jsonl')).stat().st_size,
                           sha256=hashlib.sha256((OUT/(tag+'.stdout.jsonl')).read_bytes()).hexdigest()),
               stderr=dict(bytes=len(errors),sha256=hashlib.sha256(errors).hexdigest()))
(OUT / (tag + '.execution.json')).write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt))
if status:
    print(errors.decode('utf-8', errors='replace'))
raise SystemExit(status)
