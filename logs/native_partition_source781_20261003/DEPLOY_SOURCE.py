from datetime import datetime
from pathlib import Path
import hashlib, json, paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
publication = json.loads((proof/'five_copy781.json').read_bytes())
previous = json.loads((base/'cpu_boundary_terminal780/INTAKE.json').read_bytes())
sources = dict(previous['source_sha256'])
owned = (
    'modeling/trifusion/partitioned_evidence_clip.py', 'tools/run_native_partitioned.py',
    'tools/check_partitioned_native_pair.py', 'tools/check_partitioned_backward.py',
    'tools/queue_native_partitioned.py', 'tools/report_native_partitioned.py',
    'refine-logs/native_model_partition_v5/EXPERIMENT_PLAN.md',
    'refine-logs/native_model_partition_v5/EXPERIMENT_CODE_REVIEW.md',
)
sources.update({name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in owned})
assert len(sources) == 312
target = base/'partition_deploy781'
assert not target.exists()
target.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
code = f'''from pathlib import Path
from datetime import datetime
import ast, hashlib, json, os, shutil, subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
python='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
assert head=={publication['head']!r}
expected={sources!r}
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in expected.items())
for name in {list(owned[:6])!r}:
 ast.parse((root/name).read_text())
oldjob={previous['job']!r}
assert not Path('/proc/'+str(oldjob['controller_pid'])).exists()
assert not Path('/proc/'+str(oldjob['child_pid'])).exists()
assert Path(python).is_file()
weight=root/'pertrained-model/ViT-B-16.pt'
assert hashlib.sha256(weight.read_bytes()).hexdigest()=='5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
campaign=root/'logs/independent_native_evidence_20261003_v5'
report=root/'results/native_partition_complete_20261003'
output=Path('/home/gaob/trifusion-native-evidence-v5')
assert not campaign.exists() and not report.exists() and not output.exists()
assert shutil.disk_usage(root).free>=2*1024**3
assert shutil.disk_usage(output.parent).free>=18*384*1024**2+600*1024**2+256*1024**2+2*1024**3
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
used={{int(row.split(',')[0]):int(row.split(',')[1]) for row in inventory.splitlines()}}
free_pairs=[lead for lead in (0,2) if all(used[device]<500 for device in (lead,lead+1))]
assert free_pairs,inventory
lead=free_pairs[0]
snapshot=root/'logs/native_partition_launch_source781_20261003'
assert not snapshot.exists()
snapshot.mkdir()
for name,digest in expected.items():
 dst=snapshot/'_source'/name
 dst.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(root/name,dst)
 assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
command=[python,'-B',str(root/'tools/queue_native_partitioned.py'),'--campaign',str(campaign),'--report-dir',str(report),'--gpu',str(lead)]
with (snapshot/'launcher.log').open('x') as log:
 process=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=f'{{lead}},{{lead+1}}'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={{'status':'LAUNCHED_NOT_VALIDATED','at':datetime.now().astimezone().isoformat(),
 'port':2026,'controller_pid':process.pid,'initial_physical_gpus':[lead,lead+1],
 'gpu_pairs':[[0,1],[2,3]],'max_parallel_jobs':2,'max_owned_physical_gpus':4,
 'campaign':str(campaign),'report_dir':str(report),'output_root':str(output),
 'snapshot_root':str(snapshot),'source_head':head,'source_sha256':expected,'command':command,
 'boundary':'New reviewed one-process model-placement candidate. No unchanged CPU-save restart. Nine full-batch M0 before any formal50; launch is not parity/capacity/metric PASS.'}}
(snapshot/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
rc = stdout.channel.recv_exit_status()
client.close()
(target/'stdout.json').write_bytes(data)
(target/'stderr.txt').write_bytes(error)
(target/'EXIT.json').write_text(json.dumps({'exit_code':rc,'at':datetime.now().astimezone().isoformat()})+'\n')
assert rc == 0, error.decode()
record = json.loads(data)
print(json.dumps({k:v for k,v in record.items() if k!='source_sha256'},indent=2))
