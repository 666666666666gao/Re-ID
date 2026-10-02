"""Read-only verification of the user's 2026-only TriFusion compute policy."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import json
import paramiko

out = Path('C:/Users/gb/.codex_tmp/trifusion_compute_policy754')
assert not out.exists()
out.mkdir()

def observe(target):
    port, root = target
    code = f'''
from datetime import datetime
from pathlib import Path
import json, subprocess
root = Path({root!r})
processes = subprocess.check_output(['ps','-u','gaob','-o','pid=,ppid=,args='], text=True)
project_processes = [line.strip() for line in processes.splitlines()
                     if 'trifusion' in line.lower()]
gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu',
                               '--format=csv,noheader,nounits'], text=True)
compute = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
campaign = root/'logs/training_feature_scale_20261003_v2/campaign.json'
state = json.loads(campaign.read_text()) if campaign.is_file() else None
print(json.dumps({{'observed_at':datetime.now().astimezone().isoformat(), 'port':{port},
 'project_root':str(root), 'project_processes':project_processes, 'gpu':gpu, 'compute':compute,
 'F2_campaign':{{k:state[k] for k in ('status','controller_pid','report_invocations','report_exit_code','report_completed_at')}}
               if state is not None else None,
 'boundary':'Read-only processes and GPU inventory; no queue, job, model, configuration or result changed.'}}))
'''
    client = paramiko.SSHClient()
    client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
    client.connect('172.19.12.138',port=port,username='gaob',
                   key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
    stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
    stdin.write(code)
    stdin.channel.shutdown_write()
    data, error = stdout.read(), stderr.read()
    status = stdout.channel.recv_exit_status()
    client.close()
    (out/f'{port}.stdout.txt').write_bytes(data)
    (out/f'{port}.stderr.txt').write_bytes(error)
    assert status == 0, error.decode()
    return json.loads(data)

with ThreadPoolExecutor(max_workers=2) as pool:
    observations = list(pool.map(observe,((2026,'/data/gaob/Re-ID/Trifusion'),
                                         (2025,'/data2/gb/Re-ID/Trifusion'))))
receipt = {'recorded_at':datetime.now().astimezone().isoformat(),
           'authorized_training_ports':[2026], 'authorized_physical_gpu_indices':[0,1,2,3],
           'max_simultaneous_single_gpu_jobs':4, 'new_2025_training_authorized':False,
           'observations':observations}
(out/'COMPUTE_POLICY.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2),flush=True)
