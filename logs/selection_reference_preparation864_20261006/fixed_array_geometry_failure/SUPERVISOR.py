from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys
folder=Path(__file__).resolve().parent
settings=json.loads((folder/'SETTINGS.json').read_text())
environment={**os.environ,'CUDA_VISIBLE_DEVICES':''}
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(folder/'diagnose.py'),'--seal',str(folder/'INPUT_SEAL.json'),'--output',settings['output']]
with (folder/'stdout.txt').open('wb') as out,(folder/'stderr.txt').open('wb') as err:
 child=subprocess.Popen(command,cwd='/data/gaob/Re-ID/Trifusion',env=environment,stdout=out,stderr=err)
 record=dict(pid=child.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
 (folder/'CHILD.json').write_text(json.dumps(record,indent=2)+'\n')
 exit_code=child.wait()
record.update(exit_code=exit_code,completed_at=datetime.now().astimezone().isoformat())
(folder/'EXIT.json').write_text(json.dumps(record,indent=2)+'\n')
sys.exit(exit_code)
