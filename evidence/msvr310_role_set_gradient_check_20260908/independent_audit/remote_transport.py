"""Use the authorized private SSH session without printing its contents."""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import io
import sys

OUT = Path(__file__).resolve().parent
script_path = OUT / sys.argv[1]
result_path = OUT / sys.argv[2]
helper = Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py')
namespace = {'__name__': 'independent_audit_transport'}

class PrivateCapture(io.StringIO):
    def reconfigure(self, **kwargs):
        pass

with redirect_stdout(PrivateCapture()), redirect_stderr(PrivateCapture()):
    exec(compile(helper.read_text(encoding='utf-8'), str(helper), 'exec'), namespace)
c = namespace['c']
stdin, stdout, stderr = c.exec_command('CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /root/miniconda3/envs/tri_reid/bin/python -B -u -', timeout=300)
stdin.write(script_path.read_text(encoding='utf-8'))
stdin.channel.shutdown_write()
result = stdout.read()
errors = stderr.read()
exit_code = stdout.channel.recv_exit_status()
result_path.write_bytes(result)
result_path.with_suffix('.stderr.txt').write_bytes(errors)
result_path.with_suffix('.exit_code.txt').write_text(str(exit_code)+'\n', encoding='utf-8')
c.close()
print('Read-only remote audit exit code:', exit_code)
print('Result bytes:', len(result), 'stderr bytes:', len(errors))
sys.exit(exit_code)
