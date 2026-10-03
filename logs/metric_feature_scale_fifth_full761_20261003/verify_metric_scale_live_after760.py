"""Revalidate original live handles at goal continuation; no training poll loop."""
from pathlib import Path
import json
import paramiko

private = Path('C:/Users/gb/.codex_tmp/training_feature_scale_split755')
output = private/'controller_live_after760'
assert not output.exists()
output.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
code = '''
from pathlib import Path
from datetime import datetime
import json,shutil
pid=2691826
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=json.loads((root/'logs/metric_feature_scale_20261003_v1/campaign.json').read_bytes())
command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\\0',b' ').decode()
assert '/tools/queue_metric_feature_scale.py --campaign ' in command
full=[row for row in campaign['jobs'] if row['phase']=='full']
running=[]
for row in full:
 if row['status']=='RUNNING':
  child_pid=row['pid']
  child_command=Path(f'/proc/{child_pid}/cmdline').read_bytes().replace(b'\\0',b' ').decode()
  assert '/tools/queue_metric_feature_scale.py --worker ' in child_command
  running.append({'dataset':row['dataset'],'variant':row['variant'],'gpu':row['gpu'],
                  'worker_pid':child_pid,'live':True,'worker_command':child_command})
print(json.dumps({'observed_at':datetime.now().astimezone().isoformat(),'controller_pid':pid,
 'controller_live':True,'controller_command':command,'phase':campaign['phase'],
 'status':campaign['status'],'completed_full':sum(r['status']=='COMPLETE' and r['exit_code']==0 for r in full),
 'report_invocations':campaign['report_invocations'],'running_live':running,
 'disk_free_bytes':shutil.disk_usage(root).free,
 'boundary':'One CPU read revalidating actual original process handles; no model/scorer/optimizer/report or training restart.'}))
'''
stdin,stdout,stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data,error = stdout.read(),stderr.read()
status = stdout.channel.recv_exit_status()
client.close()
(output/'stdout.txt').write_bytes(data)
(output/'stderr.txt').write_bytes(error)
assert status == 0, error.decode()
record = json.loads(data)
assert all(row['gpu'] in (0,1,2,3) for row in record['running_live'])
(output/'LIVE.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))
