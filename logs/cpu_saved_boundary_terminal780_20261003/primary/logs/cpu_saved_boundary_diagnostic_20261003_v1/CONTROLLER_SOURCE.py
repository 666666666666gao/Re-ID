from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys
campaign=Path(sys.argv[1]);path=campaign/'JOB.json';job=json.loads(path.read_text())
job['controller_pid']=os.getpid()
with (campaign/'diagnostic.log').open('x') as log:
 child=subprocess.Popen(job['command'],cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['physical_gpu'])),stdout=log,stderr=subprocess.STDOUT)
 job.update(status='RUNNING',child_pid=child.pid);path.write_text(json.dumps(job,indent=2)+'\n')
 code=child.wait()
job.update(status='FAILED' if code else 'COMPLETE',exit_code=code,completed_at=datetime.now().astimezone().isoformat())
path.write_text(json.dumps(job,indent=2)+'\n');sys.exit(code)
