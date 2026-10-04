from pathlib import Path
from datetime import datetime
import json,os,subprocess
launch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_launch_20261005_837')
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/tools/queue_deployment_metric_role.py', '--campaign', '/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_v1_20261005_837', '--report-dir', '/data/gaob/Re-ID/Trifusion/results/deployment_metric_role_v1_complete_20261005_837']
with (launch/'console.log').open('x') as log:
 result=subprocess.run(command,cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\n')
raise SystemExit(result.returncode)
