from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
scope_path=root/'refine-logs/independent_role_heads_fixed_best_diagnosis_v1/SOURCE_SCOPE.json'
seal_path=scope_path.parent/'INPUT_SEAL.json'
scope=json.loads(scope_path.read_text());seal=json.loads(seal_path.read_text())
assert head=='abe4e703b680a5fc7291b84ecf4744294c70698f'
assert seal['schema']=='trifusion-independent-role-heads-fixed-best-diagnosis-v1'
assert len(scope['source_sha256'])==388 and scope['source_sha256']==seal['source_sha256']
assert len(seal['artifact_sha256'])==241 and len(seal['rows'])==6
assert all(sha(root/n)==d for n,d in scope['source_sha256'].items())
assert all(sha(Path(n))==d for n,d in seal['artifact_sha256'].items())
original=root/'results/independent_role_heads_fixed_best_v1_20261007_875'
failure=root/'logs/independent_role_heads_fixed_best_launch_20261007_875/EXIT.json'
failed=json.loads((original/'campaign.json').read_text())
assert failed['status']=='FAILED' and failed['optimizer_updates']==0
assert [(r['dataset'],r['status'],r['exit_code']) for r in failed['jobs']]==[('RGBNT201','COMPLETE',0),('MSVR310','COMPLETE',0),('RGBNT100','FAILED',1)]
assert json.loads(failure.read_text())['exit_code']==1 and not (original/'RGBNT100_semantic').exists()
frozen={str(p):sha(p) for p in [original/'campaign.json',failure,
    original/'RGBNT201_semantic/DIAGNOSIS.json',original/'MSVR310_semantic/DIAGNOSIS.json',
    original/'RGBNT100_semantic.log']}
plans=root/'refine-logs/independent_role_heads_pending100_fixed_best_v1'
administrative_contract={str(p):sha(p) for p in (plans/'EXPERIMENT_PLAN.md',plans/'EXPERIMENT_CODE_REVIEW.md')}
old=root/'logs/independent_role_heads_launch_20261006_873'
assert json.loads((old/'EXIT.json').read_text())['exit_code']==0
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
        assert '/data/gaob/Re-ID/Trifusion/tools/' not in cmd,cmd
memory=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used',
    '--format=csv,noheader,nounits'],text=True)
used={int(line.split(',')[0]):int(line.split(',')[1]) for line in memory.splitlines()}
assert used[0]<500 and used[1]<500
free=shutil.disk_usage(root).free
assert free>=2415919104
journal=root/'logs/independent_role_heads_fixed_best_pending100_launch_20261007_876'
output=root/'results/independent_role_heads_fixed_best_pending100_20261007_876'
assert not journal.exists() and not output.exists()
journal.mkdir()
output.mkdir()
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',
    str(root/'tools/diagnose_independent_role_heads_best.py'),
    '--campaign',str(root/'logs/independent_role_heads_v1_20261006_873'),
    '--seal',str(seal_path),'--output-dir',str(output),'--dataset','RGBNT100','--variant','semantic']
qualified=dict(at=datetime.now().astimezone().isoformat(),head=head,free_bytes=free,
    storage_required_bytes=2415919104,gpu_memory_only=memory,source_count=388,
    source_scope_sha256=sha(scope_path),input_seal_sha256=sha(seal_path),
    input_artifact_count=241,command=command,original_failure_and_completed_sha256=frozen,administrative_contract_sha256=administrative_contract,
    boundary='One never-forwarded RGBNT100 original mAP-best, no training or re-selection. Only26GPU0/1; no25/power/temperature or source sync while active.')
(journal/'QUALIFIED.json').write_text(json.dumps(qualified,indent=2)+'\n')
supervisor=r"""from pathlib import Path
from datetime import datetime
import json,os,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
j=root/'logs/independent_role_heads_fixed_best_pending100_launch_20261007_876'
q=json.loads((j/'QUALIFIED.json').read_text())
with (j/'controller.log').open('x') as log:
    p=subprocess.Popen(q['command'],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (j/'CHILD.json').write_text(json.dumps(dict(pid=p.pid,
        start_ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21]),
        at=datetime.now().astimezone().isoformat(),command=q['command']))+'\n')
    status=p.wait()
(j/'EXIT.json').write_text(json.dumps(dict(exit_code=status,
    completed_at=datetime.now().astimezone().isoformat()))+'\n')
"""
compile(supervisor,'head_fixed_best_supervisor875','exec')
(journal/'supervisor.py').write_text(supervisor)
with (journal/'supervisor.log').open('x') as log:
    p=subprocess.Popen(['/usr/bin/python3','-B',str(journal/'supervisor.py')],cwd=root,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(status='FIRST_PENDING100_FIXED_BEST_SUPERVISOR_LIVE',pid=p.pid,
    start_ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21]),
    at=datetime.now().astimezone().isoformat(),head=head,physical_gpus=[0,1],
    output_dir=str(output),journal=str(journal),command=command,
    boundary='One pending100 supervisor handle verified; no original endpoint replay or new completion implied.')
(journal/'LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(dict(launch=launch,qualified=qualified,
    files={str((journal/n).relative_to(root)):sha(journal/n)
        for n in ('QUALIFIED.json','supervisor.py','LAUNCH.json')})))
