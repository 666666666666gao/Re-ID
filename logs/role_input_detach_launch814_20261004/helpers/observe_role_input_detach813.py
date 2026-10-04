from datetime import datetime
from pathlib import Path
import json
import sys
import time
import paramiko

delay = int(sys.argv[1])
assert 0 <= delay <= 300
time.sleep(delay)
root = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/role_input_detach813_observations')
root.mkdir(exist_ok=True)
packet = root / datetime.now().strftime('%H%M%S_%f')
packet.mkdir()
code = '''
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
for name in ('campaign.json','manifest.json','RGBNT201_semantic_m0.log','RGBNT201_semantic_train.log','prepare_RGBNT201_semantic.log','initialization/RGBNT201_semantic.json'):
 path=campaign/name
 if path.is_file():
  texts['campaign_'+name.replace('/','_')]=path.read_text()
for name in ('LAUNCH.json','EXIT.json','console.log','supervisor.log','supervisor.py'):
 path=launch/name
 if path.is_file():
  texts['launch_'+name]=path.read_text()
receipts={}
for phase in ('m0','full'):
 run=root/('trained-model/'+campaign.name+'_'+phase+'_semantic_RGBNT201')
 path=run/'training.json'
 if path.exists():
  training=json.loads(path.read_text())
  texts[phase+'_RGBNT201_semantic_training.json']=path.read_text()
  receipts[phase]={'epochs_completed':len(training['history']),'history':training['history'],
   'production_m0_diagnostics':training.get('production_m0_diagnostics'),'completed_at':training.get('completed_at')}
 steps=run/'training_steps.jsonl'
 if steps.is_file():
  with steps.open() as stream:
   lines=deque(stream,maxlen=1)
  receipts.setdefault(phase,{})['latest_step']=json.loads(lines[-1]) if lines else None
  receipts[phase]['step_log_bytes']=steps.stat().st_size
log=campaign/'RGBNT201_semantic_train.log'
tail=log.read_text().splitlines()[-10:] if log.is_file() else []
result={'at':datetime.now().astimezone().isoformat(),'controller':identity,'campaign':state,
 'active_process':active_identity,'exit':json.loads((launch/'EXIT.json').read_text()) if (launch/'EXIT.json').exists() else None,
 'console_tail':(launch/'console.log').read_text().splitlines()[-10:],
 'active_log_tail':tail,'receipts':receipts,'texts':texts,
 'source_sha_unchanged':all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in manifest['source_sha256'].items()),
 'disk_free_bytes':shutil.disk_usage(root).free,
 'gpu_observation':subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)}
print(json.dumps(result))
'''
(packet / 'remote_observer.py').write_text(code, encoding='utf-8')
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(30)
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_text(json.dumps({'exit_code': exit_code}) + '\n', encoding='utf-8')
client.close()
assert exit_code == 0, error.decode()
record = json.loads(data)
texts = packet / 'texts'
texts.mkdir()
for name, value in record['texts'].items():
    (texts / name).write_text(value, encoding='utf-8')
compact_receipts = {}
for phase, receipt in record['receipts'].items():
    diagnostics = receipt.get('production_m0_diagnostics')
    compact_receipts[phase] = {key: value for key, value in receipt.items() if key != 'production_m0_diagnostics'}
    if compact_receipts[phase].get('latest_step'):
        compact_receipts[phase]['latest_step'] = {key: value for key, value in compact_receipts[phase]['latest_step'].items() if key != 'lr'}
    if diagnostics:
        compact_receipts[phase]['m0'] = {key: diagnostics[key] for key in ('effective_optimizer_updates', 'author_bn_batches_tracked', 'detail_parameters')}
print(json.dumps({'packet': str(packet), 'at': record['at'], 'status': record['campaign']['status'],
 'completed_jobs': sum(r['status']=='COMPLETE' for r in record['campaign']['jobs']),
 'active': record['campaign'].get('active_command'), 'process': record['active_process'], 'exit': record['exit'],
 'console_tail': record['console_tail'], 'active_log_tail': record['active_log_tail'],
 'receipts': compact_receipts, 'source_sha_unchanged': record['source_sha_unchanged'],
 'gpu_observation': record['gpu_observation']}, ensure_ascii=False))
