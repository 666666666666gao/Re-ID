"""Private transport only. Never emits helper contents, credentials or host address."""
import argparse
import contextlib
import os
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('script')
parser.add_argument('label')
args = parser.parse_args()
script = (OUT / args.script).resolve()
assert script.is_relative_to(OUT)
namespace = {'__name__': 'private_audit_transport'}
with open(os.devnull, 'w', encoding='utf-8') as private_output, contextlib.redirect_stdout(private_output), contextlib.redirect_stderr(private_output):
    helper = Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py')
    exec(compile(helper.read_text(encoding='utf-8-sig'), str(helper), 'exec'), namespace)
client = namespace['c']
command = 'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/tri_reid/bin/python -B -u -X utf8 -'
receipt = {'started_at': datetime.now(timezone.utc).isoformat(), 'command': command, 'stdin_script': args.script, 'script_sha256': __import__('hashlib').sha256(script.read_bytes()).hexdigest(), 'transport': 'private existing Paramiko helper; credentials withheld'}
(OUT/(args.label+'.command.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
stdin, stdout, stderr = client.exec_command(command, timeout=600)
stdin.write(script.read_text(encoding='utf-8'))
stdin.channel.shutdown_write()
out = stdout.read()
err = stderr.read()
code = stdout.channel.recv_exit_status()
(OUT/(args.label+'.stdout.txt')).write_bytes(out)
(OUT/(args.label+'.stderr.txt')).write_bytes(err)
receipt.update(exit_code=code, completed_at=datetime.now(timezone.utc).isoformat(),stdout_bytes=len(out),stderr_bytes=len(err))
(OUT/(args.label+'.command.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
client.close()
print(json.dumps(receipt,indent=2))
if code:
    print(err.decode('utf-8',errors='replace'))
