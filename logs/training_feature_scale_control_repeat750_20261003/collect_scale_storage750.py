"""Read current F2 storage and process state only; never delete or run models."""
from pathlib import Path
import json
import paramiko

output = Path('C:/Users/gb/.codex_tmp/training_feature_scale_storage750')
assert not output.exists()
output.mkdir()
code = '''
from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/training_feature_scale_20261003_v2'
state=json.loads((campaign/'campaign.json').read_text())
rows=[]
for job in state['jobs']:
 folder=root/'trained-model'/('training_feature_scale_20261003_v2_'+job['phase']+'_'+job['variant']+'_'+job['dataset'])
 files={p.name:p.stat().st_size for p in sorted(folder.iterdir()) if p.is_file()} if folder.is_dir() else {}
 rows.append({'phase':job['phase'],'dataset':job['dataset'],'variant':job['variant'],
  'status':job['status'],'folder_exists':folder.is_dir(),'files_bytes':files})
pid=state['controller_pid'];proc=Path('/proc')/str(pid)
print(json.dumps({'observed_at':datetime.now().astimezone().isoformat(),
 'controller_pid':pid,'controller_live':proc.is_dir(),
 'controller_cmdline':(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode() if proc.is_dir() else None,
 'campaign_status':state['status'],'campaign_phase':state['phase'],
 'report_invocations':state['report_invocations'],'disk_free_bytes':shutil.disk_usage(root).free,
 'owned_run_rows':rows,'boundary':'CPU file-size/process snapshot only. Live file sizes may change. No deletion, hash replay, model, scoring or restart.'}))
'''
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
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
(output / 'STORAGE.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps({'observed_at':record['observed_at'],'controller_live':record['controller_live'],
                  'free_bytes':record['disk_free_bytes'],
                  'existing_run_folders':sum(row['folder_exists'] for row in record['owned_run_rows']),
                  'm0_probe_files_present':sum('m0_reload_probe.pth' in row['files_bytes'] for row in record['owned_run_rows']),
                  'owned_run_file_bytes':sum(sum(row['files_bytes'].values()) for row in record['owned_run_rows'])}))
