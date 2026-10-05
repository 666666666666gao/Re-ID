"""Observe the original five-model diagnosis near measured historical duration."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
launch=json.loads((base/'deployment_metric_five_fixed_best_launch850/stdout.json').read_bytes())
assert launch['status']=='FIVE_FIXED_BEST_DIAGNOSIS_LAUNCHED_ONCE' and launch['planned_models']==5
old=json.loads((base/'global_task_role_fixed_best_complete/stdout.json').read_bytes())['campaign']
duration=(datetime.fromisoformat(old['completed_at'])-datetime.fromisoformat(old['started_at'])).total_seconds()
excluded=next(j for j in old['jobs'] if (j['dataset'],j['variant'])==('MSVR310','native'))
excluded_seconds=(datetime.fromisoformat(excluded['completed_at'])-datetime.fromisoformat(excluded['started_at'])).total_seconds()
target=datetime.fromisoformat(launch['at'])+timedelta(seconds=duration-excluded_seconds-90)
packet=base/'deployment_metric_fixed_best_milestone852'
assert not packet.exists()
packet.mkdir()
start=dict(status='WAITING_ONE_ORIGINAL_FIVE_DIAGNOSIS',at=datetime.now().astimezone().isoformat(),
    first_observation_at=target.isoformat(),pid=launch['pid'],start_ticks=launch['start_ticks'],poll_seconds=240,
    historical_six_wall_seconds=duration,excluded_original_msvr_native_job_seconds=excluded_seconds,
    boundary='Estimate from real prior full campaign, not elapsed NN sum; no remote read before target, new model/report/queue or power-temperature action.')
(packet/'START.json').write_bytes((json.dumps(start,indent=2)+'\n').encode('utf-8'))
print(json.dumps(start),flush=True)
while datetime.now().astimezone()<target:
    time.sleep(min(240,(target-datetime.now().astimezone()).total_seconds()))
code=f'''from datetime import datetime
from pathlib import Path
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
output=Path({launch['output_dir']!r});launch=Path({launch['launch_dir']!r})
record=json.loads((launch/'LAUNCH.json').read_text())
assert record['pid']=={launch['pid']} and record['start_ticks']=={launch['start_ticks']}
state=json.loads((output/'campaign.json').read_text())
proc=Path('/proc/'+str(record['pid'])+'/stat');process_stat=proc.read_text() if proc.exists() else None
if process_stat is not None:assert int(process_stat[process_stat.rfind(')')+2:].split()[19])==record['start_ticks']
terminal=json.loads((launch/'EXIT.json').read_text()) if (launch/'EXIT.json').exists() else None
logs={{p.name:p.read_text().splitlines()[-4:] for p in output.glob('*.log')}}
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),status=state['status'],jobs=state['jobs'],
 process_stat=process_stat,exit=terminal,logs=logs,disk_free_bytes=shutil.disk_usage(root).free)))
'''
compile(code,'<one-five-diagnosis-observer>','exec')
(packet/'REMOTE_SOURCE.py').write_bytes(code.encode('utf-8'))
counter=0
while True:
    client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
    client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
    stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
    stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(300)
    data,error=stdout.read(),stderr.read();status=stdout.channel.recv_exit_status();client.close()
    counter+=1
    (packet/f'observation{counter:02d}.json').write_bytes(data)
    (packet/f'stderr{counter:02d}.txt').write_bytes(error)
    assert status==0,error.decode('utf-8')
    result=json.loads(data)
    print(json.dumps({k:result[k] for k in ('at','status','exit','disk_free_bytes')}),flush=True)
    if result['exit'] is not None:
        (packet/'FINAL.json').write_bytes(data)
        (packet/'EXIT.json').write_bytes((json.dumps(dict(exit_code=0,observations=counter,
            at=datetime.now().astimezone().isoformat()))+'\n').encode('utf-8'))
        break
    time.sleep(240)
