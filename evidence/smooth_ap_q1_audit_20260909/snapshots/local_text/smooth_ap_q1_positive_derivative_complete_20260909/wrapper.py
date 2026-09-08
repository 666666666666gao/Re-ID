from pathlib import Path
import json,subprocess,sys,time
p=Path(__file__).resolve().parent
started=time.time()
with (p/'result.json').open('wb') as out, (p/'analysis.stderr').open('wb') as err:
    child=subprocess.Popen([sys.executable,'-B',str(p/'analysis.py')],stdout=out,stderr=err)
    (p/'running.json').write_text(json.dumps(dict(wrapper_pid=__import__('os').getpid(),pid=child.pid,started_at=started)))
    code=child.wait()
(p/'exit.json').write_text(json.dumps(dict(exit_code=code,started_at=started,completed_at=time.time())))
sys.exit(code)
