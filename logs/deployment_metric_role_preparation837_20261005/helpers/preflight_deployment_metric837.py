from pathlib import Path
import json
import paramiko

destination=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/deployment_metric_preflight837')
assert not destination.exists()
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
code=r'''
from pathlib import Path
import datetime,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()=='331a3cb6cbd61d299440231c4bf2df1840623650'
gpu=subprocess.check_output(['nvidia-smi','-i','0,1','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
active=[]
for item in Path('/proc').iterdir():
    if not item.name.isdigit():continue
    command=item/'cmdline'
    if not command.exists():continue
    raw=command.read_bytes().replace(b'\x00',b' ').decode(errors='replace')
    if str(root) in raw and ('tri_reid/bin/python' in raw or 'queue_' in raw):
        active.append({'pid':int(item.name),'command':raw[:1100]})
probes=[]
for item in (root/'trained-model').glob('**/*probe*.pth'):
    probes.append({'path':item.relative_to(root).as_posix(),'bytes':item.stat().st_size})
status={name:(root/name).read_text() for name in [
 'logs/global_task_role_launch_20261004_824/EXIT_CODE',
 'logs/global_task_role_fixed_best_launch_20261004_v1/EXIT_CODE'] if (root/name).is_file()}
print(json.dumps({'observed_at':datetime.datetime.now().astimezone().isoformat(),'head':'331a3cb6cbd61d299440231c4bf2df1840623650','gpu_memory_only':gpu,'disk_free_bytes':shutil.disk_usage(root).free,'project_processes':active,'remaining_probe_candidates':probes,'terminal_receipts':status},indent=2))
'''
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(300)
data,error=stdout.read(),stderr.read();rc=stdout.channel.recv_exit_status()
destination.mkdir(parents=True)
(destination/'stdout.json').write_bytes(data);(destination/'stderr.txt').write_bytes(error)
(destination/'EXIT_CODE.txt').write_text(str(rc)+'\n')
assert rc==0,error.decode()
value=json.loads(data)
print(json.dumps(value,indent=2))
client.close()
