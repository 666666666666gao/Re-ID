from datetime import datetime
from pathlib import Path
import hashlib
import json
import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
publication = json.loads((proof / 'five_copy813.json').read_bytes())
assert publication['status'] == 'FIVE_DOC_COPIES_VERIFIED'
packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/role_input_detach813_launch')
assert not packet.exists()
packet.mkdir()
scope_path = 'refine-logs/role_input_detach_v1/SOURCE_SCOPE.json'
seal_path = 'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json'
scope_sha = hashlib.sha256((repo / scope_path).read_bytes()).hexdigest()
seal_sha = hashlib.sha256((repo / seal_path).read_bytes()).hexdigest()
root = '/data/gaob/Re-ID/Trifusion'
campaign = root + '/logs/role_input_detach_v1_20261004_813'
report = root + '/results/role_input_detach_v1_complete_20261004_813'
launch = root + '/logs/role_input_detach_launch_20261004_813'
command = ['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', root + '/tools/queue_role_input_detach.py', '--campaign', campaign, '--report-dir', report]
supervisor = '\n'.join([
    'from pathlib import Path',
    'from datetime import datetime',
    'import json,os,subprocess',
    'launch=Path(' + repr(launch) + ')',
    'command=' + repr(command),
    "with (launch/'console.log').open('x') as log:",
    '    result=subprocess.run(command,cwd=' + repr(root) + ",env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)",
    "(launch/'EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,completed_at=datetime.now().astimezone().isoformat()))+'\\n')",
    'raise SystemExit(result.returncode)',
    '',
])
compile(supervisor, 'supervisor.py', 'exec')
code = f'''
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path({root!r})
scope=root/{scope_path!r}
seal=root/{seal_path!r}
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=={publication['head']!r}
assert hashlib.sha256(scope.read_bytes()).hexdigest()=={scope_sha!r}
assert hashlib.sha256(seal.read_bytes()).hexdigest()=={seal_sha!r}
sources=json.loads(scope.read_text())['source_sha256']
assert len(sources)==322
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in sources.items())
controls=json.loads(seal.read_text())
assert len(controls['rows'])==9 and len(controls['artifact_sha256'])==61
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in controls['artifact_sha256'].items())
devices=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in devices.splitlines())
campaign=Path({campaign!r});report=Path({report!r});launch=Path({launch!r})
assert not campaign.exists() and not report.exists() and not launch.exists()
assert shutil.disk_usage(root).free >= 6*2*384*1024**2+2*1024**3
assert Path({command[0]!r}).is_file()
launch.mkdir()
supervisor={supervisor!r}
compile(supervisor,'supervisor.py','exec')
(launch/'supervisor.py').write_text(supervisor)
with (launch/'supervisor.log').open('x') as log:
    process=subprocess.Popen(['/usr/bin/python3','-B',str(launch/'supervisor.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
record={{'status':'LAUNCHED_ONCE','at':datetime.now().astimezone().isoformat(),'pid':process.pid,'start_ticks':ticks,
 'launch_dir':str(launch),'campaign_dir':str(campaign),'report_dir':str(report),'command':{command!r},
 'source_scope_sha256':{scope_sha!r},'control_seal_sha256':{seal_sha!r},'source_commit':{publication['head']!r},
 'source_count':len(sources),'unchanged_control_artifact_count':len(controls['artifact_sha256']),
 'gpu_observation':devices,'disk_free_bytes':shutil.disk_usage(root).free,
 'boundary':'Only26GPU0/1, one sequential two-card pair, six own8-update M0 then six fresh full50 and first strict reload. Existing warm environment unchanged. No power/temperature operations, parity or kernel repair. Historical nine controls unchanged, no old retired M0 verifier; scientific Goal active/unmet.'}}
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
compile(code, 'remote_launch_source.py', 'exec')
(packet / 'remote_launch_source.py').write_text(code, encoding='utf-8')
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(60)
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_text(json.dumps({'exit_code': exit_code}) + '\n', encoding='utf-8')
client.close()
assert exit_code == 0, error.decode()
print(json.dumps(json.loads(data), indent=2))
