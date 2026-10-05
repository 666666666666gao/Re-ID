from datetime import datetime
from pathlib import Path
import json,os,subprocess
launch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_fixed_best_five_launch_20261005_850')
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_fixed_best_diagnosis_v1/DIAGNOSE_FIVE.py', '--campaign', '/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_v1_20261005_837', '--seal', '/data/gaob/Re-ID/Trifusion/refine-logs/deployment_metric_fixed_best_diagnosis_v1/INPUT_SEAL.json', '--output-dir', '/data/gaob/Re-ID/Trifusion/results/deployment_metric_fixed_best_five_20261005_850']
with (launch/'console.log').open('x') as log:
 result=subprocess.run(command,cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\n')
raise SystemExit(result.returncode)
