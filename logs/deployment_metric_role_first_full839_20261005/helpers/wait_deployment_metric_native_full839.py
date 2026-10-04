"""Observe near prior measured first full50 duration, then at240s if needed."""
from datetime import datetime,timedelta
from pathlib import Path
import json,time
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet=base/'deployment_metric_native_full_milestone839';assert not packet.exists()
launch=json.loads((base/'deployment_metric_launch837/stdout.json').read_bytes())
current=dict(active_command=json.loads((base/'deployment_metric_RGBNT201_semantic_full839/CAMPAIGN_SNAPSHOT.json').read_bytes())['campaign']['active_command'])
assert current['active_command']['mode']=='train' and current['active_command']['status']=='RUNNING'
assert current['active_command']['command'][current['active_command']['command'].index('--variant')+1]=='native'
previous=json.loads((base/'global_task_role_native_full827/stdout.json').read_bytes())['files']['trained-model/global_task_role_v1_20261004_824_full_native_RGBNT201/training.json'];previous=json.loads(previous)
duration=datetime.fromisoformat(previous['completed_at'])-datetime.fromisoformat(previous['started_at'])
deadline=datetime.fromisoformat(current['active_command']['started_at'])+duration-timedelta(seconds=90)
packet.mkdir()
start=dict(status='WAITING_MEASURED_FIRST_FULL50_WINDOW',started_at=datetime.now().astimezone().isoformat(),first_observation_at=deadline.isoformat(),
 prior_measured_training_and_epoch_evaluation_seconds=duration.total_seconds(),campaign=launch['campaign'],supervisor_pid=launch['pid'],poll_seconds=240,
 boundary='One local scheduled observer nearestimatedend;then240s onlyif still running. No NN/GPU/power/temp query,duplicate queue or checkpointselection.')
(packet/'START.json').write_text(json.dumps(start,indent=2)+'\n',encoding='utf-8')
print(json.dumps(start),flush=True)
while datetime.now().astimezone()<deadline:
    time.sleep(min(240,(deadline-datetime.now().astimezone()).total_seconds()))
code=f'''from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');campaign=Path({launch['campaign']!r});parent=Path({launch['launch_dir']!r})
state=json.loads((campaign/'campaign.json').read_text())
full=next(j for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==('RGBNT201','native','full'))
terminal=json.loads((parent/'EXIT.json').read_text()) if (parent/'EXIT.json').exists() else None
pid={launch['pid']};proc=Path('/proc/'+str(pid));alive=False
if proc.exists():
 stat=(proc/'stat').read_text();alive=int(stat[stat.rfind(')')+2:].split()[19])=={launch['start_ticks']}
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),supervisor_alive=alive,parent_exit=terminal,campaign=state,first_full=full,
 disk_free_bytes=shutil.disk_usage(root).free,console_tail=(parent/'console.log').read_text()[-2500:])))
'''
compile(code,'remote_native_full_observation839.py','exec')
(packet/'remote_source.py').write_text(code,encoding='utf-8')
counter=0
while True:
    client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
    client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
    stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
    data,error=out.read(),err.read();rc=out.channel.recv_exit_status();client.close();counter+=1
    (packet/f'observation{counter:02d}.json').write_bytes(data);(packet/f'stderr{counter:02d}.txt').write_bytes(error)
    assert rc==0,error.decode()
    value=json.loads(data)
    print(json.dumps(dict(observed_at=value['observed_at'],first_full_status=value['first_full']['status'],parent_exit=value['parent_exit'],disk_free_bytes=value['disk_free_bytes'])),flush=True)
    if value['first_full']['status'] in ('COMPLETE','FAILED') or value['parent_exit'] is not None:
        (packet/'FINAL.json').write_bytes(data)
        (packet/'EXIT.json').write_text(json.dumps(dict(exit_code=0,observations=counter,completed_at=datetime.now().astimezone().isoformat()))+'\n')
        break
    time.sleep(240)
