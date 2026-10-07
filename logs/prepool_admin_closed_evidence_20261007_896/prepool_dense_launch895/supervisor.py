from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/prepool_dense_launch_20261007_895'
command=[sys.executable,'-B',str(root/'tools/queue_prepool_dense_correspondence.py'),
    '--campaign',str(root/'logs/prepool_dense_v1_20261007_895'),
    '--report-dir',str(root/'results/prepool_dense_complete_20261007_895')]
with (journal/'controller.log').open('xb') as log:
    child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT)
    (journal/'CHILD.json').write_text(json.dumps(dict(pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat()))+'\n')
    code=child.wait()
(journal/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\n')
raise SystemExit(code)
