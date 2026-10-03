from datetime import datetime
from pathlib import Path
import json
import paramiko

target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research794_observations') / datetime.now().strftime('%H%M%S')
assert not target.exists()
target.mkdir(parents=True)
code = '''
from datetime import datetime
from pathlib import Path
import json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
launch=root/'logs/native_research_v6_launch_20261003_794'
campaign=root/'logs/native_research_v6_20261003_794'
record={'at':datetime.now().astimezone().isoformat(),'head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'disk_free_bytes':shutil.disk_usage(root).free,'gpu0_1':subprocess.check_output(['nvidia-smi','--id=0,1',
        '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()}
for name in ('LAUNCH.json','CONTROLLER.json','EXIT.json'):
    path=launch/name
    record[name]=json.loads(path.read_text()) if path.exists() else None
state=campaign/'campaign.json'
value=json.loads(state.read_text()) if state.exists() else None
record['campaign']={'status':value['status'],'controller_pid':value['controller_pid'],
    'active_command':value.get('active_command'),'preparation':[{'dataset':r['dataset'],'mode':r['mode'],'variant':r.get('variant'),
    'status':r.get('status'),'exit_code':r.get('exit_code')} for r in value['preparation']],
    'jobs':[{key:r.get(key) for key in ('dataset','variant','phase','status','exit_code','completed_at')} for r in value['jobs']]} if value else None
pids=[record['LAUNCH.json']['supervisor_pid']] if record['LAUNCH.json'] else []
if record['CONTROLLER.json']: pids.append(record['CONTROLLER.json']['pid'])
if value and value.get('active_command'): pids.append(value['active_command']['pid'])
processes=[]
for pid in pids:
    proc=Path('/proc')/str(pid)
    row={'pid':pid,'exists':proc.exists()}
    if proc.exists():
        stat=(proc/'stat').read_text().split()
        row.update(state=stat[2],start_ticks=stat[21],command_line=(proc/'cmdline').read_bytes().decode().replace(chr(0),' '))
    processes.append(row)
record['processes']=processes
names=['controller.log','supervisor.log']
record['log_tails']={name:(launch/name).read_text()[-6000:] for name in names if (launch/name).exists()}
if value and value.get('active_command'):
    row=value['active_command']
    active_log=(f"prepare_{row['dataset']}_{row['variant']}.log" if row['mode']=='prepare' else
                f"initial_forward_pair_{row['dataset']}.log" if row['mode']=='initial_forward_pair' else
                f"{next(r['dataset'] for r in value['jobs'] if row in r.get('steps',[]))}_{next(r['variant'] for r in value['jobs'] if row in r.get('steps',[]))}_{row['mode']}.log")
    record['log_tails'][active_log]=(campaign/active_log).read_text()[-6000:]
record['completed_m0_receipts']=[]
for job in value['jobs'] if value else []:
    if job['phase']=='m0' and job['status']=='COMPLETE':
        data=job['result']
        record['completed_m0_receipts'].append({'dataset':job['dataset'],'variant':job['variant'],'status':data['status'],
          'steps':data['history'][0]['steps'],'reload_max_abs_difference':data['m0']['reload_max_abs_difference'],
          'production_updates':data['production_m0_diagnostics']['effective_optimizer_updates'],
          'author_bn_batches_tracked':data['production_m0_diagnostics']['author_bn_batches_tracked']})
print(json.dumps(record))
'''
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(30)
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(target / 'stdout.json').write_bytes(data)
(target / 'stderr.txt').write_bytes(error)
(target / 'EXIT.json').write_text(json.dumps({'exit_code': exit_code}) + '\n')
client.close()
assert exit_code == 0, error.decode()
print(data.decode())
