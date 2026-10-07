from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/fixed_five_incremental_best_launch_20261007_892'
campaign=root/'logs/fixed_five_incremental_best_diagnosis_20261007_892'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
exit=json.loads((journal/'EXIT.json').read_text())
state=json.loads((campaign/'campaign.json').read_text())
assert state['status'] in ('FAILED','COMPLETE')
assert (exit['exit_code']==0)==(state['status']=='COMPLETE')
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and (proc/'cmdline').is_file():
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in (proc/'cmdline').read_bytes().split(bytes([0])))
seal_path=root/'refine-logs/incremental_role_objective_v1/FIXED_FIVE_DIAGNOSIS_INPUT_SEAL.json'
seal=json.loads(seal_path.read_text())
assert len(seal['source_sha256'])==424 and len(seal['artifact_sha256'])==75
assert all(sha(root/n)==d for n,d in seal['source_sha256'].items())
assert all(sha(Path(n))==d for n,d in seal['artifact_sha256'].items())
rows=[]
for job in state['jobs']:
    if job['status']=='COMPLETE':
        p=campaign/(job['dataset']+'_'+job['objective'])/'DIAGNOSIS.json'
        report=json.loads(p.read_text())
        assert report['status']=='COMPLETE' and report['model_state_before_sha256']==report['model_state_after_sha256']
        for name,item in report['artifacts'].items():
            assert (p.parent/name).stat().st_size==item['bytes'] and sha(p.parent/name)==item['sha256']
        row=next(r for r in seal['rows'] if r['dataset']==job['dataset'] and r['variant']==job['objective'])
        assert report['input_checkpoint_sha256']==row['checkpoint_sha256']
        assert all(abs(report['scores']['fused']['metrics'][k]-v)<1e-5 for k,v in row['metrics'].items())
        rows.append(report)
if exit['exit_code']==0:
    assert len(rows)==5
    assert sum(row['diagnostic'][s]['global_norm']['count'] for row in rows for s in ('query','gallery'))==16926
files={}
for directory in (journal,campaign):
    for p in sorted(directory.rglob('*')):
        if p.is_file() and p.suffix in ('.json','.py','.log','.txt'):
            files[p.relative_to(root).as_posix()]=dict(text=p.read_text(),sha256=sha(p),bytes=p.stat().st_size)
files[seal_path.relative_to(root).as_posix()]=dict(text=seal_path.read_text(),sha256=sha(seal_path),bytes=seal_path.stat().st_size)
print(json.dumps(dict(status='FIXED_FIVE_COMPONENT_DIAGNOSIS_TERMINAL_RECEIVED',at=datetime.now().astimezone().isoformat(),
    original_exit=exit,campaign=state,rows=rows,source_count=424,artifact_count=75,
    current_inputs_physically_unchanged=True,free_bytes=shutil.disk_usage(root).free,files=files,
    boundary='Original selected best weights only, no optimizer/NN rerun/new checkpoint/gain/threshold. Tensor artifacts remain26. Exact component retrieval is post-selection descriptive and not an independent training ablation. Failed100repair remains missing.')))
