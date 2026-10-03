from datetime import datetime
from pathlib import Path
import json
import paramiko

target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/goal_execution_state793')
assert not target.exists()
target.mkdir()
code = '''
from datetime import datetime
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
names=('queue_independent_native_evidence.py','run_independent_native_evidence.py','check_independent_native_pair.py',
       'run_native_backward_boundaries.py','native_backward_boundary_capture.py','check_native_backward_boundaries.py')
rows=[]
for line in subprocess.check_output(['ps','-u','gaob','-o','pid=,args='],text=True).splitlines():
    parts=line.strip().split(None,1)
    if len(parts)!=2:
        continue
    matched=[name for name in names if str(root/'tools'/name) in parts[1]]
    if matched:
        rows.append({'pid':int(parts[0]),'project_tool_names':matched})
texts=('refine-logs/CURRENT_GOAL.md','refine-logs/independent_native_evidence_v1/EXPERIMENT_PLAN.md',
       'refine-logs/independent_native_evidence_v1/EXPERIMENT_TRACKER.md','tools/queue_independent_native_evidence.py')
record={'observed_at':datetime.now().astimezone().isoformat(),
        'head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'matching_native_campaign_or_withdrawn_diagnostic_processes':rows,
        'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in texts},
        'disk_free_bytes':shutil.disk_usage(root).free,
        'model_or_training_started':False,'boundary':'Read-only process/source/disk observation; no GPU/model/parity/scorer/weight operation.'}
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
(target / 'EXIT.json').write_text(json.dumps({'at':datetime.now().astimezone().isoformat(),'exit_code':exit_code})+'\n',encoding='utf-8')
client.close()
assert exit_code == 0, error.decode()
record=json.loads(data)
print(json.dumps(record,indent=2))
