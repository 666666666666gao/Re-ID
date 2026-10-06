from pathlib import Path
from datetime import datetime
import json,os,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
launch=root/'logs/signal_selection_reference_launch_20261006_866'
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/queue_signal_selection_reference.py'),'--campaign',str(root/'logs/signal_selection_reference_v1_20261006_866'),'--report-dir',str(root/'results/signal_selection_reference_complete_20261006_866')]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
with (launch/'stdout.txt').open('x') as output,(launch/'stderr.txt').open('x') as error:
 child=subprocess.Popen(command,cwd=root,env=env,stdout=output,stderr=error)
 row=dict(pid=child.pid,start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),started_at=datetime.now().astimezone().isoformat(),command=command,physical_gpus=[0,1])
 (launch/'CHILD.json').write_text(json.dumps(row,indent=2)+'\n')
 code=child.wait()
(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()),indent=2)+'\n')
raise SystemExit(code)
