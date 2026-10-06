from pathlib import Path
from datetime import datetime
import ast, hashlib, json, os, shutil, subprocess, sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_signal_selection_reference as panel
journal=root/'logs/independent_role_head_preparation_20261006_872'
assert not journal.exists(); journal.mkdir()
def sha(p): return panel.sha(p)
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()=='7e684e00b0f29f00dd847c14864abf33676c6759'
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
        assert '/data/gaob/Re-ID/Trifusion/tools/' not in cmd,cmd
panel.require_sources(); panel.require_protected()
files=['modeling/trifusion/independent_role_heads.py','tools/run_independent_role_heads.py',
       'tools/check_independent_role_heads.py','tools/queue_independent_role_heads.py','tools/report_independent_role_heads.py']
for n in files: ast.parse((root/n).read_text())
with (journal/'component_stdout.log').open('x') as log:
    result=subprocess.run(['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',
        str(root/'tools/check_independent_role_heads.py'),'--output',str(journal/'CPU_WITNESS.json')],
        cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
(journal/'CPU_EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode,at=datetime.now().astimezone().isoformat()))+'\n')
assert result.returncode==0,(journal/'component_stdout.log').read_text()
seal=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
campaign=root/'logs/signal_selection_reference_v1_20261006_870'
state=json.loads((campaign/'campaign.json').read_text());assert state['status']=='COMPLETE'
cpu=root/'logs/selection_report_completion_20261006_871'
assert json.loads((cpu/'EXIT.json').read_text())['exit_code']==0
for name in ('LAUNCH.json','CHILD.json'):
    r=json.loads((cpu/name).read_text());p=Path('/proc')/str(r['pid'])
    assert not p.exists() or int((p/'stat').read_text().split()[21])!=r['start_ticks']
matrix=json.loads((campaign/'accepted_matrix.json').read_text())
origins=json.loads((campaign/'endpoint_origins.json').read_text())
metadata=json.loads((campaign/'batch_metadata_paths.json').read_text())
targets=[r for r in matrix['rows'] if r['dataset']=='RGBNT201' or (r['dataset']=='RGBNT100' and r['variant']=='global_only')]
assert len(targets)==4
keys=('mAP','Rank-1','Rank-5','Rank-10'); proofs=[]
for r in targets:
    key=f'{r["dataset"]}:{r["variant"]}'
    assert panel.accepted_row(Path(origins[key]),r['dataset'],r['variant'],batch_metadata_path=Path(metadata[key]))==r
    p=Path(r['run_dir'])/'best_map.pth'
    assert p.resolve().is_relative_to((root/'trained-model').resolve()) and p.stat().st_nlink==1
    assert str(p) not in seal['artifact_sha256'] and str(p.relative_to(root)) not in seal['artifact_sha256']
    winner=next(x for x in seal['rows'] if (x['dataset'],x['variant'])==(r['dataset'],'global_only'))
    assert all(winner['metrics'][k]>=r['metrics'][k] for k in keys)
    keep=Path(winner['run_dir'])/'best_map.pth'
    assert sha(keep)==winner['checkpoint_sha256'] and sha(p)==r['checkpoint_sha256']
    proofs.append(dict(target=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)),
                       original_accepted_row=r,retained_winner=winner))
qualified=dict(at=datetime.now().astimezone().isoformat(),free_before=shutil.disk_usage(root).free,
    proofs=proofs,source_sha256={n:sha(root/n) for n in files},
    protected_control_seal_sha256=sha(root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'),
    boundary='Only four closed, all-CMC-dominated self-trained selection-reference bests. Source-reference binary replay for these four retires; original histories, distances and receipts remain. No author, public, RAW187, nondominated or current dependency removed.')
(journal/'QUALIFIED.json').write_text(json.dumps(qualified,indent=2)+'\n')
with (journal/'RETIREMENT.jsonl').open('x') as f:
    for proof in proofs:
        p=Path(proof['target']['path']); assert sha(p)==proof['target']['sha256'];p.unlink()
        f.write(json.dumps(dict(retired_at=datetime.now().astimezone().isoformat(),**proof['target']))+'\n');f.flush()
assert all(not Path(p['target']['path']).exists() for p in proofs)
panel.require_sources();panel.require_protected()
complete=dict(status='CPU_HEAD_WITNESS_AND_FOUR_QUALIFIED_RETIREMENTS_COMPLETE',
    at=datetime.now().astimezone().isoformat(),retired=4,retired_bytes=sum(p['target']['bytes'] for p in proofs),
    free_after=shutil.disk_usage(root).free,required_start_bytes=4*384*1024**2+2*1024**3,
    source_sha256=qualified['source_sha256'],cpu_witness=json.loads((journal/'CPU_WITNESS.json').read_text()))
(journal/'COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n')
print(json.dumps(dict(**complete,files={str(p.relative_to(root)):sha(p) for p in journal.iterdir() if p.is_file()})))
