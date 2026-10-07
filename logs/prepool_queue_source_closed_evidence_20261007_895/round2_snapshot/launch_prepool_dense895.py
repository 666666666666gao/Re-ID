from pathlib import Path
import json
import paramiko

proof = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002/four_copy895_2025_pending.json')
publication = json.loads(proof.read_bytes())
packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/prepool_dense_launch895')
assert not packet.exists()
packet.mkdir()
supervisor = '''from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/prepool_dense_launch_20261007_895'
command=[sys.executable,'-B',str(root/'tools/queue_prepool_dense_correspondence.py'),
    '--campaign',str(root/'logs/prepool_dense_v1_20261007_895'),
    '--report-dir',str(root/'results/prepool_dense_complete_20261007_895')]
with (journal/'controller.log').open('xb') as log:
    child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT)
    (journal/'CHILD.json').write_text(json.dumps(dict(pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat()))+'\\n')
    code=child.wait()
(journal/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\\n')
raise SystemExit(code)
'''
code = '''from pathlib import Path
from datetime import datetime,timedelta
import ast,hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/prepool_dense_launch_20261007_895'
campaign=root/'logs/prepool_dense_v1_20261007_895'
report=root/'results/prepool_dense_complete_20261007_895'
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==HEAD
assert not journal.exists() and not campaign.exists() and not report.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
scope_path=root/'refine-logs/prepool_dense_correspondence_v1/SOURCE_SCOPE.json'
scope=json.loads(scope_path.read_text())['source_sha256']
assert len(scope)==428 and all(sha(root/name)==digest for name,digest in scope.items())
review=json.loads((root/'refine-logs/prepool_dense_correspondence_v1/QUEUE_SOURCE_REVIEW.json').read_text())
assert review['verdict']=='PASS' and review['scope']=='SOURCE_ONLY'
seal_path=root/'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json'
seal=json.loads(seal_path.read_text())
assert len(seal['rows'])==6 and len(seal['artifact_sha256'])==45
assert all(sha(Path(name))==digest for name,digest in seal['artifact_sha256'].items())
assert json.loads((root/'logs/qualified_five_incremental_full_launch_20261007_888/EXIT.json').read_text())['exit_code']==0
assert json.loads((root/'logs/fixed_five_incremental_best_launch_20261007_892/EXIT.json').read_text())['exit_code']==0
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and (proc/'cmdline').is_file():
        assert not any(word.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for word in (proc/'cmdline').read_bytes().split(bytes([0])))
memory=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
free=shutil.disk_usage(root).free
assert free>=3758096384
journal.mkdir(parents=True)
(journal/'QUALIFIED.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),head=HEAD,
    source_count=428,input_count=45,free_bytes=free,storage_required_bytes=3758096384,
    physical_01_memory=memory,source_scope_sha256=sha(scope_path),input_seal_sha256=sha(seal_path),
    boundary='Sources/text reviewed; actual M0 pending. Existing queue waits physical0/1 memory<500 before each neural child, even if busy at this snapshot. No25/2/3/power/temp/install.'),indent=2)+'\\n')
(journal/'supervisor.py').write_text(SUPERVISOR)
with (journal/'supervisor.log').open('xb') as log:
    process=subprocess.Popen(['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(journal/'supervisor.py')],
        cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
started=datetime.now().astimezone()
launch=dict(supervisor_pid=process.pid,start_ticks=(Path('/proc')/str(process.pid)/'stat').read_text().split()[21],
    started_at=started.isoformat(),first_observation_at=(started+timedelta(minutes=3)).isoformat(),
    expected_total_hours=[4,6],campaign=str(campaign),journal=str(journal),report=str(report),head=HEAD,
    phase='THREE_PER_ENDPOINT_M0_THEN_FRESH_FULL50',total_expected_epochs=150,total_expected_updates=6484,
    estimate_basis='Previous matched3 full50s about3.1h plus extra no-grad teacher/geometry and preparation; estimate not measured. No mid-score method changes.')
(journal/'LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\\n')
print(json.dumps(launch))
'''
code = 'HEAD = ' + repr(publication['head']) + '\n' + code.replace('SUPERVISOR', repr(supervisor))
compile(code, 'prepool_dense895_launch_remote', 'exec')
(packet/'SOURCE.py').write_text(code,encoding='utf-8')
(packet/'supervisor.py').write_text(supervisor,encoding='utf-8')
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(300)
data,error=stdout.read(),stderr.read()
status=stdout.channel.recv_exit_status()
client.close()
(packet/'REMOTE.json').write_bytes(data)
(packet/'STDERR.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps({'exit_code':status})+'\n')
assert status==0,error.decode()
print(data.decode())
