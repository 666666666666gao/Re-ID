from pathlib import Path
from datetime import datetime
import json,os,subprocess
launch=Path('/data/gaob/Re-ID/Trifusion/logs/native_fixed_best_launch_20261004_811')
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '/data/gaob/Re-ID/Trifusion/tools/diagnose_native_research_best.py', '--campaign', '/data/gaob/Re-ID/Trifusion/logs/native_research_v6_20261003_794', '--seal', '/data/gaob/Re-ID/Trifusion/refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json', '--output-dir', '/data/gaob/Re-ID/Trifusion/results/native_fixed_best_20261004_811']
with (launch/'console.log').open('x') as log:
 result=subprocess.run(command,cwd='/data/gaob/Re-ID/Trifusion',env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\n')
raise SystemExit(result.returncode)
