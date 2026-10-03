from pathlib import Path
from datetime import datetime
import json
import paramiko

root=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/gpu_limits796')/('sudo_check_'+datetime.now().strftime('%H%M%S'))
root.mkdir()
code='''
from datetime import datetime
from pathlib import Path
import json,subprocess
def run(command):
    r=subprocess.run(command,capture_output=True,text=True)
    return {'command':command,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
query=['nvidia-smi','--id=0,1','--query-gpu=index,uuid,temperature.gpu,power.draw,power.limit,fan.speed,utilization.gpu','--format=csv,noheader,nounits']
before=run(query)
assert before['exit_code']==0,before
limits=[float(line.split(',')[4]) for line in before['stdout'].splitlines()]
assert len(limits)==2
record={'at':datetime.now().astimezone().isoformat(),'before':before}
if max(limits)>250:
    record['noninteractive_privileged_set']=run(['sudo','-n','nvidia-smi','--id=0,1','--power-limit=250'])
record['after']=run(query)
record['supported_target_temperature']=run(['nvidia-smi','--id=0,1','-q','-d','SUPPORTED_GPU_TARGET_TEMP'])
record['process_states']={str(pid):(Path('/proc')/str(pid)/'stat').read_text().split()[2] for pid in (3854540,3863278)}
print(json.dumps(record))
'''
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(30)
data,error=stdout.read(),stderr.read()
exit_code=stdout.channel.recv_exit_status()
client.close()
(root/'stdout.json').write_bytes(data)
(root/'stderr.txt').write_bytes(error)
(root/'EXIT.json').write_text(json.dumps({'exit_code':exit_code})+'\n')
assert exit_code==0,error.decode()
print(data.decode())
