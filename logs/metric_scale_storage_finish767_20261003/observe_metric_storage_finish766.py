"""Read-only observer for the separately recorded storage finish, every240s."""
from pathlib import Path
from datetime import datetime,timedelta
import json,os,time,paramiko

base=Path('C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766')
launch=json.loads((base/'deploy_v2/LAUNCH.json').read_bytes())
assert launch['port']==2026 and launch['gpu']==1
out=base/'finish_observer';assert not out.exists();out.mkdir()
due=datetime.fromisoformat(launch['observed_at'])+timedelta(seconds=240)
record={'status':'WAITING_FIRST_STORAGE_FINISH_MILESTONE','pid':os.getpid(),
 'controller_pid':launch['controller_pid'],'due':due.isoformat(),'poll_seconds':240,
 'boundary':'Read-only new storage-finish observer. No original controller/report/model/scorer/training restart.'}
def save():
 record['updated_at']=datetime.now().astimezone().isoformat()
 (out/'STATUS.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
save()
while datetime.now().astimezone()<due:
 time.sleep(min(240,(due-datetime.now().astimezone()).total_seconds()))
index=0
while True:
 index+=1
 client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
 client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
 code=f'''
from pathlib import Path
from datetime import datetime
import json
finish=Path({launch['finish_directory']!r});status=finish/'STATUS.json'
asset=Path({launch['asset_directory']!r});log=asset/'launcher.log'
actual=json.loads(status.read_text()) if status.exists() else None
live=(Path('/proc')/str({launch['controller_pid']})).exists()
print(json.dumps({{'observed_at':datetime.now().astimezone().isoformat(),'controller_live':live,
 'finish':actual,'launcher_log_tail':log.read_text()[-5000:] if log.exists() else None}}))
'''
 stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
 stdin.write(code);stdin.channel.shutdown_write()
 data,error=stdout.read(),stderr.read();exit_code=stdout.channel.recv_exit_status()
 (out/f'{index:04d}.stdout.json').write_bytes(data);(out/f'{index:04d}.stderr.txt').write_bytes(error)
 client.close()
 record.update(observation_number=index,observer_exit_code=exit_code)
 assert exit_code==0,error.decode()
 actual=json.loads(data);record.update(last_receipt=str(out/f'{index:04d}.stdout.json'),
  controller_live=actual['controller_live'],finish=actual['finish'])
 if not actual['controller_live']:
  record['status']='STORAGE_FINISH_TERMINAL_OBSERVED'
  save();break
 record['status']='WAITING_LIVE_STORAGE_FINISH';save();time.sleep(240)
print(json.dumps(record,indent=2),flush=True)
