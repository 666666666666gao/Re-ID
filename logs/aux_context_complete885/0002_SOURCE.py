HEAD = 'a9341bd681999b846ce0de41ba480f95c501ceb4'
from pathlib import Path
from datetime import datetime,timedelta
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion');journal=root/'logs/initial_aux_autocast_context_launch_20261007_884';output=root/'results/initial_aux_autocast_context_20261007_884/DIAGNOSIS.json'
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==HEAD
assert not journal.exists() and not output.parent.exists()
scope_path=root/'refine-logs/incremental_role_objective_v1/AUTOCAST_CONTEXT_SOURCE_SCOPE.json'
source=json.loads(scope_path.read_text())['source_sha256'];assert len(source)==406
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in source.items())
seal_path=root/'refine-logs/incremental_role_objective_v1/INITIAL_BOUNDARY_INPUT_SEAL.json';seal=json.loads(seal_path.read_text());assert len(seal['artifact_sha256'])==48
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==d for n,d in seal['artifact_sha256'].items())
assert json.loads((root/'logs/incremental_role_objective_m0_launch_20261007_878/EXIT.json').read_text())['exit_code']==1
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        words=(p/'cmdline').read_bytes().split(bytes([0]))
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in words),words
memory=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert all(int(row.split(',')[1])<500 for row in memory.splitlines())
free=shutil.disk_usage(root).free;assert free>=2*1024**3+256*1024**2
journal.mkdir(parents=True)
(journal/'QUALIFIED.json').write_text(json.dumps(dict(head=HEAD,at=datetime.now().astimezone().isoformat(),
    free_bytes=free,physical_01_memory=memory,source_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
    input_seal_sha256=hashlib.sha256(seal_path.read_bytes()).hexdigest(),source_count=406,input_count=48,
    original_gate_fail_preserved=True),indent=2)+'\n')
(journal/'supervisor.py').write_text("from pathlib import Path\nfrom datetime import datetime\nimport json,os,subprocess,sys\nroot=Path('/data/gaob/Re-ID/Trifusion');journal=root/'logs/initial_aux_autocast_context_launch_20261007_884'\ncommand=[sys.executable,'-B',str(root/'tools/diagnose_initial_aux_autocast_context.py'),'--output',str(root/'results/initial_aux_autocast_context_20261007_884/DIAGNOSIS.json')]\nwith (journal/'controller.log').open('xb') as log:\n    child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT)\n    (journal/'CHILD.json').write_text(json.dumps(dict(pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat()))+'\\n')\n    code=child.wait()\n(journal/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(code)\n")
with (journal/'supervisor.log').open('xb') as log:
    p=subprocess.Popen(['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(journal/'supervisor.py')],cwd=root,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
started=datetime.now().astimezone();value=dict(supervisor_pid=p.pid,started_at=started.isoformat(),
    start_ticks=(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],head=HEAD,
    first_observation_at=(started+timedelta(minutes=3)).isoformat(),journal=str(journal),output=str(output),
    optimizer_updates=0,planned_forwards=1,planned_isolated_vjps=2,phase='INITIAL_AUX_AUTOCAST_CONTEXT_DIAGNOSTIC')
(journal/'LAUNCH.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
