from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys
root=Path('/data/gaob/Re-ID/Trifusion');journal=root/'logs/incremental_role_objective_m0_launch_20261007_886'
command=[sys.executable,'-B',str(root/'tools/queue_incremental_checked_m0.py'),'--campaign',str(root/'logs/incremental_role_objective_m0_v1_20261007_886')]
with (journal/'controller.log').open('xb') as log:
    child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT)
    (journal/'CHILD.json').write_text(json.dumps(dict(pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat()))+'\n')
    code=child.wait()
(journal/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\n')
raise SystemExit(code)
