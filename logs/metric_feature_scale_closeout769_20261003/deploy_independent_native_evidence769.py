"""Launch the reviewed immutable nine-arm queue on2026; never launch on2025."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/deploy769')
assert not target.exists()
publication = json.loads((proof / 'five_copy769.json').read_bytes())
review_path = repo / 'refine-logs/independent_native_evidence_v1/EXPERIMENT_CODE_REVIEW.json'
review = json.loads(review_path.read_bytes())
assert not review['blocking_findings'] and review['verdict'].upper() in ('PASS', 'WARN')
claims = json.loads((repo / 'refine-logs/metric_feature_scale_v1/CLAIMS_FROM_RESULTS.json').read_bytes())
audit = json.loads((repo / 'refine-logs/metric_feature_scale_v1/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['integrity_status'] in ('pass', 'warn') and not audit['blocking_findings']
assert claims['integrity_status'] in ('pass', 'warn') and claims['claim_supported'] == 'no'
names = [
    'modeling/trifusion/image_native_evidence.py',
    'modeling/trifusion/independent_native_roles.py',
    'modeling/trifusion/evidence_author_heads.py',
    'tools/run_independent_native_evidence.py',
    'tools/check_independent_native_pair.py',
    'tools/queue_independent_native_evidence.py',
    'tools/report_independent_native_evidence.py',
    'refine-logs/independent_native_evidence_v1/EXPERIMENT_PLAN.md',
    'refine-logs/independent_native_evidence_v1/EXPERIMENT_CODE_REVIEW.md',
]
expected = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in names}
target.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
code = f'''from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()=={publication['head']!r}
expected={expected!r}
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in expected.items())
old=json.loads((root/'logs/metric_feature_scale_20261003_v1/campaign.json').read_text())
assert old['status']=='COMPLETE' and old['report_exit_code']==0 and old['report_invocations']==1
free=shutil.disk_usage(root).free
assert free>=18*384*1024**2+600*1024**2+256*1024**2+2*1024**3
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
available=[int(row.split(',')[0]) for row in memory.splitlines() if int(row.split(',')[0]) in range(4) and int(row.split(',')[1])<500]
assert available
gpu=available[0]
campaign=root/'logs/independent_native_evidence_20261003_v1'
report=root/'results/independent_native_evidence_complete_20261003'
launchlog=root/'logs/independent_native_evidence_20261003_v1.launcher.log'
assert not campaign.exists() and not report.exists() and not launchlog.exists()
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/queue_independent_native_evidence.py'),'--campaign',str(campaign),'--report-dir',str(report),'--gpu',str(gpu)]
with launchlog.open('x') as log:
 process=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'status':'INITIALIZER_QUEUE_LAUNCHED','pid':process.pid,'port':2026,'physical_gpu_scope':[0,1,2,3],'max_parallel':4,'initialization_gpu':gpu,'command':command,'launcher_log':str(launchlog),'disk_free_bytes_before':free,'head':{publication['head']!r},'source_sha256':expected,'launched_at':datetime.now().astimezone().isoformat(),'boundary':'Initial construction, parity and all9 M0 precede formal training. No2025 execution. Launch is not M0 or scientific success.'}}))
'''
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(target / 'stdout.json').write_bytes(data)
(target / 'stderr.txt').write_bytes(error)
(target / 'EXIT.json').write_text(json.dumps({'exit_code': exit_code,
    'collected_at': datetime.now().astimezone().isoformat()}) + '\n')
client.close()
assert exit_code == 0, error.decode()
print(data.decode(), end='')
