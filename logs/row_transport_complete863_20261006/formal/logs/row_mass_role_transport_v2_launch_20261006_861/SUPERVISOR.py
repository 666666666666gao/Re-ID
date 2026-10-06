from pathlib import Path
from datetime import datetime
import subprocess,json,os,sys
root=Path('/data/gaob/Re-ID/Trifusion')
launch=root/'logs/row_mass_role_transport_v2_launch_20261006_861'
command=[sys.executable,'-B',str(root/'tools/queue_row_mass_role_transport.py'),
 '--campaign',str(root/'logs/row_mass_role_transport_v2_20261006_860'),
 '--report-dir',str(root/'results/row_mass_role_transport_v2_complete_20261006_860')]
with (launch/'console.log').open('x') as log:
 child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT)
 (launch/'CHILD.json').write_text(json.dumps(dict(pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat()))+'\n')
 code=child.wait()
(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\n')
sys.exit(code)
