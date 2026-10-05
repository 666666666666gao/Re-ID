from datetime import datetime
from pathlib import Path
import json,os,subprocess
launch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_pending_fixed_best_launch_20261005_854')
with (launch/'console.log').open('x') as log:
 result=subprocess.run(['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_pending_fixed_best_v1/CONTINUE_RGBNT100.py'],cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\n')
raise SystemExit(result.returncode)
