"""Observe the sole campaign near final training completion, through its first report."""
import ast
from datetime import datetime,timedelta
import json
from pathlib import Path
import time

import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet=base/'semantic_capacity_campaign_complete_milestone856'
assert not packet.exists()
packet.mkdir()
last=json.loads((base/'semantic_capacity_rgb100_start_milestone856/FINAL.json').read_bytes())
full=next(r for r in last['campaign']['jobs'] if (r['dataset'],r['phase'])==('RGBNT100','full'))
train=next(s for s in full['steps'] if s['mode']=='train')
start=datetime.fromisoformat(train['started_at'])
history=json.loads((base/'global_task_role_rgb100_native_full829/received/trained-model/global_task_role_v1_20261004_824_full_native_RGBNT100/training.json').read_bytes())
seconds=(datetime.fromisoformat(history['completed_at'])-datetime.fromisoformat(history['started_at'])).total_seconds()
target=start+timedelta(seconds=seconds-180)
record=dict(status='WAITING_EXISTING_FINAL_FULL50_AND_FIRST_REPORT',at=datetime.now().astimezone().isoformat(),
 first_observation_at=target.isoformat(),actual_final_train_started_at=start.isoformat(),
 matched_raw_native_training_seconds=seconds,early_margin_seconds=180,poll_seconds=240,
 boundary='Estimate only. Existing controller, first strict evaluation and its automatic CPU report. No early remote progress, additional report, restart, source change or power/temperature action.')
(packet/'START.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps(record),flush=True)
tree=ast.parse(Path('C:/Users/gb/.codex_tmp/wait_semantic_capacity_first_full856.py').read_text(encoding='utf-8'))
assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
code=ast.literal_eval(assignment.value)
compile(code,'capacity_complete_observer.py','exec')
(packet/'REMOTE_SOURCE.py').write_bytes(code.encode())
while datetime.now().astimezone()<target:
 time.sleep(min(240,(target-datetime.now().astimezone()).total_seconds()))
index=0
while True:
 client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
 client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
 stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
 stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(60)
 data,error=stdout.read(),stderr.read();status=stdout.channel.recv_exit_status();client.close();index+=1
 (packet/f'observation{index:02}.json').write_bytes(data);(packet/f'stderr{index:02}.txt').write_bytes(error)
 assert status==0,error.decode()
 value=json.loads(data);state=value['campaign']
 summary=dict(at=value['at'],campaign_status=state['status'],parent_exit=value['exit'],
  active=state.get('active_command',{}).get('mode'),accepted=list(value['acceptance']),
  report_invocations=state['report_invocations'],report_exit_code=state.get('report_exit_code'),
  disk_free_bytes=value['disk_free_bytes'])
 print(json.dumps(summary),flush=True)
 if value['exit'] is not None:
  (packet/'FINAL.json').write_bytes(data)
  (packet/'EXIT.json').write_bytes((json.dumps(dict(exit_code=0,observed=summary,observer_completed_at=datetime.now().astimezone().isoformat()))+'\n').encode())
  break
 time.sleep(240)
