HEAD = 'b61a1b2e61e42bab0d9197bb87806c18c8daa502'
from pathlib import Path
from datetime import datetime,timedelta
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion'); base=root/'refine-logs/incremental_role_objective_v1'
journal=root/'logs/fixed_five_incremental_best_launch_20261007_892'
campaign=root/'logs/fixed_five_incremental_best_diagnosis_20261007_892'
seal_path=base/'FIXED_FIVE_DIAGNOSIS_INPUT_SEAL.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==HEAD
assert not journal.exists() and not campaign.exists() and not seal_path.exists()
scope_path=base/'FIXED_FIVE_DIAGNOSIS_SOURCE_SCOPE.json'
scope=json.loads(scope_path.read_text())['source_sha256']; assert len(scope)==424
assert all(sha(root/n)==d for n,d in scope.items())
review=json.loads((base/'FIXED_FIVE_DIAGNOSIS_SOURCE_REVIEW_20261007_892.json').read_text())
assert review['verdict']=='PASS' and review['blocking_count']==0
assert review['reviewed_code_sha256']==sha(root/'tools/diagnose_incremental_fixed_best.py')
terminal=json.loads((root/'logs/qualified_five_incremental_full_launch_20261007_888/EXIT.json').read_text())
assert terminal['exit_code']==0
report=json.loads((root/'results/qualified_five_incremental_full_complete_20261007_888/SUMMARY.json').read_text())
assert report['status']=='COMPLETE'
received=json.loads((base/'FIVE_FORMAL_COMPLETE_20261007_891.json').read_text())
assert received['formal_completed']==5 and received['actual_parent_exit']==terminal
control_path=base/'INPUT_SEAL.json'; controls=json.loads(control_path.read_text())
initial_path=base/'INITIAL_BOUNDARY_INPUT_SEAL.json'; initial=json.loads(initial_path.read_text())
assert len(controls['artifact_sha256'])==45 and len(initial['artifact_sha256'])==48
artifacts=dict(controls['artifact_sha256'])
for n,d in initial['artifact_sha256'].items():
    assert n not in artifacts or artifacts[n]==d
    artifacts[n]=d
for row in received['rows']:
    run=Path(row['run_dir'])
    for name,key in (('best_map.pth','checkpoint_sha256'),('official_distances.pt','distance_sha256'),('official_metrics.json','receipt_sha256')):
        n=str(run/name); d=row[key]
        assert n not in artifacts or artifacts[n]==d
        artifacts[n]=d
manifest=root/'logs/qualified_five_incremental_full_v1_20261007_888/manifest.json'
for n,d in json.loads(manifest.read_text())['initialization_sha256'].items():
    assert n not in artifacts or artifacts[n]==d
    artifacts[n]=d
for p in (root/'pertrained-model/ViT-B-16.pt',control_path,initial_path,manifest,
          root/'logs/qualified_five_incremental_full_launch_20261007_888/EXIT.json',
          root/'results/qualified_five_incremental_full_complete_20261007_888/SUMMARY.json',
          base/'FIVE_FORMAL_COMPLETE_20261007_891.json'):
    n=str(p); d=sha(p)
    assert n not in artifacts or artifacts[n]==d
    artifacts[n]=d
assert all(Path(n).is_absolute() and sha(Path(n))==d for n,d in artifacts.items())
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and (proc/'cmdline').is_file():
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in (proc/'cmdline').read_bytes().split(bytes([0])))
memory=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert len(memory.splitlines())==2 and all(int(row.split(',')[1])<500 for row in memory.splitlines())
free=shutil.disk_usage(root).free; assert free>2*1024**3
seal=dict(schema='trifusion-incremental-fixed-best-diagnosis-v1',registered_at=datetime.now().astimezone().isoformat(),
    rows=received['rows'],global_rows=[r for r in controls['rows'] if r['variant']=='global_only'],
    source_sha256=scope,artifact_sha256=artifacts,original_parent_exit=terminal,
    boundary='Exactly five already-selected full50 mAP-best checkpoints. Zero fitting/updates/selection. Original failed100repair remains missing. Previous45/48/source420 unchanged, new424 source-qualified. Same fixed camera/scene protocol and original thresholds. Only physical0/1 maxone split model.')
assert len(seal['rows'])==5 and len(seal['global_rows'])==3
seal_path.write_text(json.dumps(seal,indent=2)+'\n')
journal.mkdir(parents=True)
(journal/'QUALIFIED.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),head=HEAD,
    source_count=424,artifact_count=len(artifacts),free_bytes=free,physical_01_memory=memory,
    source_scope_sha256=sha(scope_path),seal_sha256=sha(seal_path),original_parent_exit=terminal),indent=2)+'\n')
(journal/'supervisor.py').write_text("from pathlib import Path\nfrom datetime import datetime\nimport json,os,subprocess,sys\nroot=Path('/data/gaob/Re-ID/Trifusion'); journal=root/'logs/fixed_five_incremental_best_launch_20261007_892'\ncommand=[sys.executable,'-B',str(root/'tools/diagnose_incremental_fixed_best.py'),\n    '--seal',str(root/'refine-logs/incremental_role_objective_v1/FIXED_FIVE_DIAGNOSIS_INPUT_SEAL.json'),\n    '--output-dir',str(root/'logs/fixed_five_incremental_best_diagnosis_20261007_892')]\nwith (journal/'controller.log').open('xb') as log:\n    child=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT)\n    (journal/'CHILD.json').write_text(json.dumps(dict(pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat()))+'\\n')\n    code=child.wait()\n(journal/'EXIT.json').write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\\n')\nraise SystemExit(code)\n")
with (journal/'supervisor.log').open('xb') as log:
    p=subprocess.Popen(['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(journal/'supervisor.py')],cwd=root,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
started=datetime.now().astimezone()
launch=dict(supervisor_pid=p.pid,start_ticks=(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],
    started_at=started.isoformat(),first_observation_at=(started+timedelta(minutes=8)).isoformat(),
    expected_minutes=[10,25],journal=str(journal),campaign=str(campaign),head=HEAD,
    phase='FIVE_SELECTED_BEST_COMPONENT_DIAGNOSIS',optimizer_updates=0,forward_records=16926,
    source_count=424,artifact_count=len(artifacts),boundary='No Git/source/server synchronization or cleanup of current five dependencies while NN active. One original controller and observer; first failure terminal, no retry.')
(journal/'LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(launch))
