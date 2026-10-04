"""One scheduled observation near the estimated first real M0 completion."""
from datetime import datetime,timedelta
from pathlib import Path
import hashlib,json,time
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
launch=json.loads((base/'deployment_metric_launch837/stdout.json').read_bytes())
packet=base/'deployment_metric_first_m0_milestone838';assert not packet.exists();packet.mkdir()
deadline=datetime.fromisoformat(launch['at'])+timedelta(minutes=4)
(packet/'START.json').write_text(json.dumps(dict(status='WAITING_ESTIMATED_FIRST_REAL_M0_MILESTONE',started_at=datetime.now().astimezone().isoformat(),deadline=deadline.isoformat(),campaign=launch['campaign'],supervisor_pid=launch['pid'],boundary='Local timer, one remote observation at estimatedM0 end;no GPU/temperature/power queries before target,no duplicate launch.'))+'\n')
print((packet/'START.json').read_text(),flush=True)
while datetime.now().astimezone()<deadline:
    time.sleep(min(60,(deadline-datetime.now().astimezone()).total_seconds()))
code=f'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');campaign=Path({launch['campaign']!r});parent=Path({launch['launch_dir']!r})
pid={launch['pid']};proc=Path('/proc/'+str(pid))
alive=False
if proc.exists():
 stat=(proc/'stat').read_text();alive=int(stat[stat.rfind(')')+2:].split()[19])=={launch['start_ticks']}
state=json.loads((campaign/'campaign.json').read_text()) if (campaign/'campaign.json').exists() else None
files={{}}
for name in ['manifest.json','campaign.json','initialization/RGBNT201_semantic.json','prepare_RGBNT201_semantic.log','RGBNT201_semantic_m0.log']:
 path=campaign/name
 if path.is_file():files[str(path.relative_to(root))]=dict(text=path.read_text(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
for path in [parent/'EXIT.json',parent/'console.log',root/'trained-model/deployment_metric_role_v1_20261005_837_m0_semantic_RGBNT201/training.json']:
 if path.is_file():files[str(path.relative_to(root))]=dict(text=path.read_text(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),supervisor_alive_matching_startticks=alive,campaign_state=state,files=files,disk_free_bytes=shutil.disk_usage(root).free,power_temperature_queried=False)))
'''
compile(code,'remote_m0_observer838.py','exec')
(packet/'remote_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n')
client.close();assert rc==0,error.decode()
value=json.loads(data);state=value['campaign_state']
print(json.dumps(dict(status='SCHEDULED_OBSERVATION_RETURNED',observed_at=value['observed_at'],supervisor_alive=value['supervisor_alive_matching_startticks'],campaign_status=state['status'] if state else None,
 jobs=[dict(dataset=j['dataset'],variant=j['variant'],phase=j['phase'],status=j['status']) for j in state['jobs'] if j['status']!='PENDING'] if state else [],disk_free_bytes=value['disk_free_bytes'])),flush=True)
