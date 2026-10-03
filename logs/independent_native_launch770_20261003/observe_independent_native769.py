"""Read-only observations of the single launched 2026 queue; no restarts."""
from datetime import datetime
from pathlib import Path
import json
import time
import paramiko

target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/observer769')
launch = json.loads(Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/deploy770/stdout.json').read_bytes())
assert launch['port'] == 2026
assert not target.exists()
target.mkdir()
code = '''from pathlib import Path
from datetime import datetime
import json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/independent_native_evidence_20261003_v1'
state=json.loads((campaign/'campaign.json').read_text())
jobs=state['jobs']
tails={}
for job in jobs:
 if job['status'] in ('RUNNING','FAILED'):
  child=campaign/(job['phase']+'_'+job['variant']+'_'+job['dataset'])
  for p in child.glob('*.log'):
   tails[str(p.relative_to(root))]=p.read_text(errors='replace').splitlines()[-12:]
if state['status']=='INITIALIZING':
 logs=sorted(campaign.glob('*.log'),key=lambda p:p.stat().st_mtime)
 if logs:
  p=logs[-1];tails[str(p.relative_to(root))]=p.read_text(errors='replace').splitlines()[-12:]
pid=state['controller_pid']
alive=Path('/proc/'+str(pid)).exists()
counts={phase:{status:sum(j['phase']==phase and j['status']==status for j in jobs) for status in ('PENDING','RUNNING','COMPLETE','FAILED')} for phase in ('m0','full')}
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'status':state['status'],'phase':state.get('phase'),'controller_pid':pid,'controller_present':alive,'counts':counts,'report_invocations':state.get('report_invocations',0),'report_exit_code':state.get('report_exit_code'),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),'disk_free_bytes':shutil.disk_usage(root).free,'initialization_count':len(list((campaign/'initialization').glob('*.json'))),'initial_pair_count':len(list(campaign.glob('initial_forward_pair_*.json'))),'log_tails':tails,'boundary':'Read-only only2026; no model import, report/scorer, restart or process control.'}))
'''
time.sleep(240)
for serial in range(1, 41):
    client = paramiko.SSHClient()
    client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
    client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
    stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
    stdin.write(code)
    stdin.channel.shutdown_write()
    data, error = stdout.read(), stderr.read()
    exit_code = stdout.channel.recv_exit_status()
    prefix = f'observation_{serial:04d}'
    (target / (prefix + '.stdout.json')).write_bytes(data)
    (target / (prefix + '.stderr.txt')).write_bytes(error)
    (target / (prefix + '.EXIT.json')).write_text(json.dumps({'exit_code': exit_code, 'at': datetime.now().astimezone().isoformat()}) + '\n')
    client.close()
    assert exit_code == 0, error.decode()
    value = json.loads(data)
    closed = not value['controller_present'] or (value['status'] == 'COMPLETE' and value['report_exit_code'] is not None)
    (target / 'STATUS.json').write_text(json.dumps({'observer_status': 'CLOSED' if closed else 'RUNNING', 'last_observation': serial, 'observation': value}, indent=2) + '\n')
    print(json.dumps({k:v for k,v in value.items() if k != 'log_tails'}), flush=True)
    if closed:
        break
    # Initial construction/M0 need engineering checks. Full50 takes much longer.
    time.sleep(240 if value['phase'] != 'full' else 1800)
