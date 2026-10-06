HEAD = '534fc4a4760889d133a1c732c091ed1035aac85b'
from pathlib import Path
from datetime import datetime,timedelta
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion');campaign=root/'logs/incremental_role_objective_m0_v1_20261007_878';journal=root/'logs/incremental_role_objective_m0_launch_20261007_878'
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==HEAD
assert not campaign.exists() and not journal.exists()
scope_path=root/'refine-logs/incremental_role_objective_v1/SOURCE_SCOPE.json'
scope=json.loads(scope_path.read_text())['source_sha256'];assert len(scope)==393
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in scope.items())
seal_path=root/'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json';seal=json.loads(seal_path.read_text())
assert len(seal['rows'])==6 and len(seal['artifact_sha256'])==45
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==d for n,d in seal['artifact_sha256'].items())
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        words=(p/'cmdline').read_bytes().split(bytes([0]))
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in words),words
memory=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert all(int(row.split(',')[1])<500 for row in memory.splitlines())
free=shutil.disk_usage(root).free;assert free>=2*1024**3+512*1024**2
journal.mkdir(parents=True)
qualified=dict(at=datetime.now().astimezone().isoformat(),head=HEAD,source_count=len(scope),input_count=45,
    source_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),input_seal_sha256=hashlib.sha256(seal_path.read_bytes()).hexdigest(),
    free_bytes=free,physical_01_memory=memory,scope='Only26physical0/1/max1 split model. Six prepare and eight-step M0 gates only; no automatic formal50.')
(journal/'QUALIFIED.json').write_text(json.dumps(qualified,indent=2)+'\n')
(journal/'supervisor.py').write_text("from pathlib import Path\nfrom datetime import datetime\nimport json,os,subprocess,sys\nroot=Path('/data/gaob/Re-ID/Trifusion');journal=root/'logs/incremental_role_objective_m0_launch_20261007_878'\ncommand=[sys.executable,'-B',str(root/'tools/queue_incremental_role_objective_m0.py'),'--campaign',str(root/'logs/incremental_role_objective_m0_v1_20261007_878')]\nwith (journal/'controller.log').open('xb') as log:\n    child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT)\n    (journal/'CHILD.json').write_text(json.dumps(dict(pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat()))+'\\n')\n    code=child.wait()\n(journal/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(code)\n")
with (journal/'supervisor.log').open('xb') as log:
    p=subprocess.Popen(['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(journal/'supervisor.py')],cwd=root,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
started=datetime.now().astimezone()
launch=dict(supervisor_pid=p.pid,start_ticks=(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],
    started_at=started.isoformat(),first_observation_at=(started+timedelta(minutes=8)).isoformat(),
    expected_total_minutes=[8,18],campaign=str(campaign),journal=str(journal),head=HEAD,phase='M0_ONLY')
(journal/'LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(launch))
