"""One read-only observation after the isolated CPU witness; no replay."""
from pathlib import Path
import json
import paramiko

root = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
output = root/'original_f3_live_after_cpu'
assert not output.exists()
output.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
code = '''
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/metric_feature_scale_20261003_v1'
state=json.loads((campaign/'campaign.json').read_bytes())
manifest=json.loads((campaign/'manifest.json').read_bytes())
source=manifest['source_sha256']
assert len(source)==265
assert source=={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source}
pid=state['controller_pid']
command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\\0',b' ').decode()
assert pid==2691826 and 'queue_metric_feature_scale.py --campaign' in command
full=[row for row in state['jobs'] if row['phase']=='full']
running=[]
for row in full:
 if row['status']=='RUNNING':
  worker=row['pid']
  args=Path(f'/proc/{worker}/cmdline').read_bytes().replace(b'\\0',b' ').decode()
  assert 'queue_metric_feature_scale.py --worker' in args
  assert row['gpu'] in (0,1,2,3)
  run=root/'trained-model'/f"metric_feature_scale_20261003_v1_full_{row['variant']}_{row['dataset']}"
  training=json.loads((run/'training.json').read_bytes())
  running.append({'dataset':row['dataset'],'variant':row['variant'],'gpu':row['gpu'],
                  'worker_pid':worker,'process_live':True,'completed_epochs':len(training['history'])})
assert len(running)<=4
print(json.dumps({'observed_at':datetime.now().astimezone().isoformat(),'training_port':2026,
 'controller_pid':pid,'controller_live':True,'status':state['status'],'phase':state['phase'],
 'completed_full':sum(r['status']=='COMPLETE' and r['exit_code']==0 for r in full),
 'running':running,'report_invocations':state['report_invocations'],
 'source_files_verified_unchanged':len(source),'disk_free_bytes':shutil.disk_usage(root).free,
 'boundary':'CPU metadata/source read only; no model, scorer, report or training/restart.'}))
'''
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
client.close()
(output/'stdout.txt').write_bytes(data)
(output/'stderr.txt').write_bytes(error)
assert status == 0, error.decode()
record = json.loads(data)
(output/'LIVE.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))
