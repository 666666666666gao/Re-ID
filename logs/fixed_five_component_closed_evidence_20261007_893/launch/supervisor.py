from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys
root=Path('/data/gaob/Re-ID/Trifusion'); journal=root/'logs/fixed_five_incremental_best_launch_20261007_892'
command=[sys.executable,'-B',str(root/'tools/diagnose_incremental_fixed_best.py'),
    '--seal',str(root/'refine-logs/incremental_role_objective_v1/FIXED_FIVE_DIAGNOSIS_INPUT_SEAL.json'),
    '--output-dir',str(root/'logs/fixed_five_incremental_best_diagnosis_20261007_892')]
with (journal/'controller.log').open('xb') as log:
    child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT)
    (journal/'CHILD.json').write_text(json.dumps(dict(pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat()))+'\n')
    code=child.wait()
(journal/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\n')
raise SystemExit(code)
