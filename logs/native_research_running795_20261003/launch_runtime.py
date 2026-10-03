from pathlib import Path
import json
import paramiko

proof = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research794_launch')
assert not target.exists()
head = json.loads((proof / 'five_copy794.json').read_bytes())['head']
code = '''
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==EXPECTED_HEAD
scope=json.loads((root/'refine-logs/native_research_v6/SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(scope)==314
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in scope.items())
used=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
assert all(int(row.split(',')[2])<500 for row in used.splitlines()),used
assert shutil.disk_usage(root).free>=12*1024**3
launch=root/'logs/native_research_v6_launch_20261003_794'
campaign=root/'logs/native_research_v6_20261003_794'
report=root/'results/native_research_v6_complete_20261003_794'
assert not launch.exists() and not campaign.exists() and not report.exists()
launch.mkdir()
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/queue_native_research.py'),
         '--campaign',str(campaign),'--report-dir',str(report)]
supervisor = """from datetime import datetime
from pathlib import Path
import json,os,subprocess
launch=Path(LAUNCH)
with (launch/'controller.log').open('x') as log:
    process=subprocess.Popen(COMMAND,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT)
    (launch/'CONTROLLER.json').write_text(json.dumps({'pid':process.pid,'command':COMMAND,'started_at':datetime.now().astimezone().isoformat()}))
    code=process.wait()
(launch/'EXIT.json').write_text(json.dumps({'exit_code':code,'completed_at':datetime.now().astimezone().isoformat()}))
raise SystemExit(code)
"""
supervisor=supervisor.replace('LAUNCH',repr(str(launch))).replace('COMMAND',repr(command)).replace('ROOT',repr(str(root)))
path=launch/'supervise.py'
compile(supervisor,str(path),'exec')
path.write_text(supervisor)
with (launch/'supervisor.log').open('x') as log:
    process=subprocess.Popen(['/usr/bin/python3','-B',str(path)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'head':EXPECTED_HEAD,'at':datetime.now().astimezone().isoformat(),'supervisor_pid':process.pid,'command':command,
        'launch':str(launch),'campaign':str(campaign),'report_dir':str(report),'gpu0_1_before':used.splitlines(),
        'disk_free_bytes_before':shutil.disk_usage(root).free,'physical_gpus':[0,1],
        'scope':'User-selected per-endpoint fresh research training; no historical parity retry or repair claim.'}
(launch/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''.replace('EXPECTED_HEAD', repr(head))
compile(code, '<remote-launch>', 'exec')
target.mkdir()
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
(target / 'EXIT.json').write_text(json.dumps({'exit_code': exit_code}) + '\n')
client.close()
assert exit_code == 0, error.decode()
print(data.decode())
