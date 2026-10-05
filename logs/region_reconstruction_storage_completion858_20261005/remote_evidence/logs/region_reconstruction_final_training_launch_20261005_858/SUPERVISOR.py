from datetime import datetime
import json,os
from pathlib import Path
import subprocess,sys
root=Path('/data/gaob/Re-ID/Trifusion');launch=Path(__file__).resolve().parent
command=[sys.executable,'-B',str(root/'logs/region_reconstruction_final_training_completion_20261005_858/ADMIN_COMPLETE858.py')]
with (launch/'console.log').open('x') as output:
 code=subprocess.run(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=output,stderr=subprocess.STDOUT).returncode
(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\n')
raise SystemExit(code)
