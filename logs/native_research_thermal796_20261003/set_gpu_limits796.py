from pathlib import Path
from datetime import datetime
import json
import paramiko

root = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/gpu_limits796') / datetime.now().strftime('%H%M%S')
root.mkdir(parents=True)
remote = '''
from datetime import datetime
import json,subprocess
def run(command):
    result=subprocess.run(command,capture_output=True,text=True)
    return {'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
query=['nvidia-smi','--id=0,1','--query-gpu=index,uuid,name,temperature.gpu,power.draw,power.limit,power.default_limit,power.min_limit,power.max_limit,utilization.gpu','--format=csv,noheader,nounits']
result={'at':datetime.now().astimezone().isoformat(),'identity':run(['id']), 'before':run(query)}
result['set_power_250']=run(['nvidia-smi','--id=0,1','--power-limit=250'])
result['after']=run(query)
result['temperature_controls']=run(['nvidia-smi','--help'])
print(json.dumps(result))
'''
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(remote)
stdin.channel.shutdown_write()
stdout.channel.settimeout(30)
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(root / 'stdout.json').write_bytes(data)
(root / 'stderr.txt').write_bytes(error)
(root / 'EXIT.json').write_text(json.dumps({'exit_code':exit_code}) + '\n')
client.close()
assert exit_code == 0, error.decode()
record = json.loads(data)
help_text = record.pop('temperature_controls')
record['temperature_control_help'] = [line for line in help_text['stdout'].splitlines() if any(key in line.lower() for key in ('target', 'temperature', 'clock', 'power limit'))]
print(json.dumps(record))
