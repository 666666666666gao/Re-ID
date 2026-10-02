from pathlib import Path
import json
import paramiko

output = Path('C:/Users/gb/.codex_tmp/training_feature_scale_wall_timing754')
assert not output.exists()
output.mkdir()
code = '''
from pathlib import Path
from datetime import datetime,timedelta
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/training_feature_scale_20261003_v2'
controller=json.loads((campaign/'campaign.json').read_text())
pid=controller['controller_pid']
proc=Path('/proc')/str(pid)
rows=[]
for variant,dataset in (('raw','RGBNT100'),):
 path=root/'trained-model'/('training_feature_scale_20261003_v2_full_'+variant+'_'+dataset)/'training.json'
 data=path.read_bytes()
 record=json.loads(data)
 modified=path.stat().st_mtime
 history=record['history']
 assert history and [row['epoch'] for row in history]==list(range(1,len(history)+1))
 started=datetime.fromisoformat(record['started_at'])
 last_complete=datetime.fromtimestamp(modified,started.tzinfo)
 seconds=(last_complete-started).total_seconds()
 mean=seconds/len(history)
 assert mean>0
 rows.append({'dataset':dataset,'variant':variant,'status':record['status'],
  'training_started_at':record['started_at'],'completed_epochs':len(history),
  'training_json_mtime':last_complete.isoformat(),'completed_epoch_wall_seconds':seconds,
  'mean_complete_epoch_wall_seconds':mean,
  'mean_training_loop_seconds':sum(row['seconds'] for row in history)/len(history),
  'estimated_training_completion_at':(last_complete+timedelta(seconds=(50-len(history))*mean)).isoformat(),
  'initial_model_state_sha256':record['initializer']['initial_model_state_sha256'],
  'public_clip_sha256':record['initializer']['public_clip_sha256'],
  'history_scores_used_for_estimate':False,
  'boundary':'Observed start/file-mtime and completed epochs; includes epoch scoring and best saves, excludes construction, final strict evaluation and queue delay. ETA only, not completion proof.'})
print(json.dumps({'observed_at':datetime.now().astimezone().isoformat(),
 'controller_pid':pid,'controller_live':proc.is_dir(),
 'controller_cmdline':(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode(),
 'phase':controller['phase'],'status':controller['status'],
 'report_invocations':controller['report_invocations'],'rows':rows,
 'disk_free_bytes':shutil.disk_usage(root).free,
 'boundary':'CPU read/arithmetic only. No model/scorer/optimizer/report invocation; no performance-based changes.'}))
'''
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
client.close()
(output / 'stdout.txt').write_bytes(data)
(output / 'stderr.txt').write_bytes(error)
assert exit_code == 0, error.decode()
record = json.loads(data)
(output / 'WALL_TIMING.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record, indent=2))
