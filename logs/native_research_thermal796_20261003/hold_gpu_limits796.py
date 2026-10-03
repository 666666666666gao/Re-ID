from pathlib import Path
from datetime import datetime
import json
import paramiko

local = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/gpu_limits796') / ('hold_' + datetime.now().strftime('%H%M%S'))
local.mkdir(parents=True)
remote = '''
from pathlib import Path
from datetime import datetime
import json,os,signal,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
launch=root/'logs/native_research_v6_launch_20261003_794'
campaign=root/'logs/native_research_v6_20261003_794'
controller=json.loads((launch/'CONTROLLER.json').read_text())['pid']
def process(pid):
    p=Path('/proc')/str(pid)
    stat=(p/'stat').read_text().split()
    return {'pid':pid,'state':stat[2],'start_ticks':stat[21],'command':(p/'cmdline').read_bytes().decode().replace(chr(0),' ')}
c=process(controller)
assert 'tools/queue_native_research.py' in c['command'] and 'native_research_v6_20261003_794' in c['command'], c
assert (Path('/proc')/str(controller)).stat().st_uid == os.getuid()
os.kill(controller,signal.SIGSTOP)
value=json.loads((campaign/'campaign.json').read_text())
active=value.get('active_command')
assert active and value['status']=='RUNNING', value['status']
p=process(active['pid'])
assert 'native_research_v6_20261003_794' in p['command'] and '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python' in p['command'], p
assert (Path('/proc')/str(active['pid'])).stat().st_uid == os.getuid()
os.kill(active['pid'],signal.SIGSTOP)
record={'at':datetime.now().astimezone().isoformat(),'reason':'User temperature <=75 C and power <=250 W target; GPU0 observed 78 C; hardware power change refused.',
        'controller_before':c,'active_before':p,'controller_after':process(controller),'active_after':process(active['pid']),
        'active_command':active,'hardware':subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,uuid,temperature.gpu,power.draw,power.limit','--format=csv,noheader,nounits'],text=True).splitlines(),
        'compute_apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines()}
(launch/'THERMAL_HOLD_796.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(remote)
stdin.channel.shutdown_write()
stdout.channel.settimeout(30)
data, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
(local/'stdout.json').write_bytes(data)
(local/'stderr.txt').write_bytes(error)
(local/'EXIT.json').write_text(json.dumps({'exit_code':status})+'\n')
client.close()
assert status == 0, error.decode()
print(data.decode())
