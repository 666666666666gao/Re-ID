from pathlib import Path
from datetime import datetime
import json,os,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
j=root/'logs/independent_role_heads_launch_20261006_873'
q=json.loads((j/'QUALIFIED.json').read_text())
with (j/'controller.log').open('x') as log:
    p=subprocess.Popen(q['command'],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (j/'CHILD.json').write_text(json.dumps(dict(pid=p.pid,start_ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21]),
        at=datetime.now().astimezone().isoformat(),command=q['command']))+'\n')
    status=p.wait()
(j/'EXIT.json').write_text(json.dumps(dict(exit_code=status,completed_at=datetime.now().astimezone().isoformat()))+'\n')
