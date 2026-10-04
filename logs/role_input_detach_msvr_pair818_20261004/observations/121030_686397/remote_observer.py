
from collections import deque
from datetime import datetime
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/role_input_detach_v1_20261004_813'
launch=root/'logs/role_input_detach_launch_20261004_813'
record=json.loads((launch/'LAUNCH.json').read_text())
proc=Path('/proc/'+str(record['pid'])+'/stat')
identity=None
if proc.exists():
 stat=proc.read_text();parts=stat[stat.rfind(')')+2:].split()
 identity={'state':parts[0],'start_ticks':int(parts[19]),'same_process':int(parts[19])==record['start_ticks']}
state=json.loads((campaign/'campaign.json').read_text())
manifest=json.loads((campaign/'manifest.json').read_text())
active=state.get('active_command')
active_identity=None
if active:
 path=Path('/proc/'+str(active['pid']))
 if path.exists():
  stat=(path/'stat').read_text();parts=stat[stat.rfind(')')+2:].split()
  active_identity={'state':parts[0],'start_ticks':int(parts[19]),'command':(path/'cmdline').read_bytes().decode().replace(chr(0),' ').strip()}
texts={}
for name in ('campaign.json','manifest.json','MSVR310_native_m0.log','MSVR310_native_train.log','prepare_MSVR310_native.log','initialization/MSVR310_native.json'):
 path=campaign/name
 if path.is_file():
  texts['campaign_'+name.replace('/','_')]=path.read_text()
for name in ('LAUNCH.json','EXIT.json','console.log','supervisor.log','supervisor.py'):
 path=launch/name
 if path.is_file():
  texts['launch_'+name]=path.read_text()
receipts={}
for phase in ('m0','full'):
 run=root/('trained-model/'+campaign.name+'_'+phase+'_native_MSVR310')
 path=run/'training.json'
 if path.exists():
  training=json.loads(path.read_text())
  texts[phase+'_MSVR310_native_training.json']=path.read_text()
  receipts[phase]={'epochs_completed':len(training['history']),'history':training['history'],
   'production_m0_diagnostics':training.get('production_m0_diagnostics'),'completed_at':training.get('completed_at')}
 steps=run/'training_steps.jsonl'
 if steps.is_file():
  with steps.open() as stream:
   lines=deque(stream,maxlen=1)
  receipts.setdefault(phase,{})['latest_step']=json.loads(lines[-1]) if lines else None
  receipts[phase]['step_log_bytes']=steps.stat().st_size
log=campaign/'MSVR310_native_train.log'
tail=log.read_text().splitlines()[-10:] if log.is_file() else []
result={'at':datetime.now().astimezone().isoformat(),'controller':identity,'campaign':state,
 'active_process':active_identity,'exit':json.loads((launch/'EXIT.json').read_text()) if (launch/'EXIT.json').exists() else None,
 'console_tail':(launch/'console.log').read_text().splitlines()[-10:],
 'active_log_tail':tail,'receipts':receipts,'texts':texts,
 'source_sha_unchanged':all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in manifest['source_sha256'].items()),
 'disk_free_bytes':shutil.disk_usage(root).free,
 'gpu_observation':subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)}
print(json.dumps(result))
