"""Warm-reuse inventory and own new CPU component witness, no training."""
from pathlib import Path
import hashlib
import json
import shlex
import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/region_reconstruction_preparation858')
packet.mkdir()
root = '/data/gaob/Re-ID/Trifusion'
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
code = '''from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
controls=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
protected=set(controls['artifact_sha256'])
capacity=json.loads((root/'logs/semantic_capacity_first_eval_completion_20261005_857/accepted_matrix.json').read_text())
weights=[]
for p in sorted((root/'trained-model').glob('*/best_map.pth')):
    receipt=p.parent/'official_metrics.json'
    training=p.parent/'training.json'
    if not receipt.is_file() or not training.is_file():continue
    value=json.loads(receipt.read_text());history=json.loads(training.read_text())
    weights.append(dict(path=str(p),bytes=p.stat().st_size,metrics=value.get('metrics'),
        selected_epoch=value.get('selected_epoch'),status=history.get('status'),
        epochs=len(history.get('history',[])),protected=str(p) in protected))
print(json.dumps(dict(head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
    free_bytes=shutil.disk_usage(root).free,
    gpu_memory=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True),
    weights=weights,capacity_rows=capacity['rows'])))
'''
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code); stdin.channel.shutdown_write(); stdout.channel.settimeout(60)
data, error = stdout.read(), stderr.read()
assert stdout.channel.recv_exit_status() == 0, error.decode()
(packet/'INVENTORY.json').write_bytes(data)
inventory = json.loads(data)
assert inventory['head'] == 'a513b681c612c6d801d505919f332861b9dcacae'
names = ['modeling/trifusion/region_evidence_reconstruction.py', 'tools/run_region_reconstruction.py',
         'tools/check_region_reconstruction.py', 'tools/queue_region_reconstruction.py',
         'tools/report_region_reconstruction.py']
sftp = client.open_sftp()
for name in names:
    sftp.put(str(repo/name), root+'/'+name)
sftp.close()
command = ['env', 'CUDA_VISIBLE_DEVICES=', '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B',
           root+'/tools/check_region_reconstruction.py', '--output',
           root+'/logs/region_reconstruction_component858/COMPONENT.json']
_, stdout, stderr = client.exec_command(shlex.join(command))
stdout.channel.settimeout(60)
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(packet/'component.stdout').write_bytes(data)
(packet/'component.stderr').write_bytes(error)
(packet/'EXECUTION.json').write_text(json.dumps(dict(exit_code=exit_code, command=command,
    source_sha256={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in names}),indent=2)+'\n')
if exit_code == 0:
    sftp = client.open_sftp()
    sftp.get(root+'/logs/region_reconstruction_component858/COMPONENT.json', str(packet/'COMPONENT.json'))
    sftp.close()
client.close()
print(json.dumps(dict(exit_code=exit_code, free_bytes=inventory['free_bytes'],
    gpu_memory=inventory['gpu_memory'], formal_weights=len(inventory['weights']), packet=str(packet))))
if exit_code:print(error.decode())
assert exit_code == 0
