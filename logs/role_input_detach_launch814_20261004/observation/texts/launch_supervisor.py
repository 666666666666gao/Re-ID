from pathlib import Path
from datetime import datetime
import json,os,subprocess
launch=Path('/data/gaob/Re-ID/Trifusion/logs/role_input_detach_launch_20261004_813')
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/tools/queue_role_input_detach.py', '--campaign', '/data/gaob/Re-ID/Trifusion/logs/role_input_detach_v1_20261004_813', '--report-dir', '/data/gaob/Re-ID/Trifusion/results/role_input_detach_v1_complete_20261004_813']
with (launch/'console.log').open('x') as log:
    result=subprocess.run(command,cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\n')
raise SystemExit(result.returncode)
