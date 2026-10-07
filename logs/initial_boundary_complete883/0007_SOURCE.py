from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and (proc/'cmdline').is_file():
        assert not any(word.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for word in (proc/'cmdline').read_bytes().split(bytes([0])))
scope=json.loads((root/'refine-logs/incremental_role_objective_v1/INITIAL_BOUNDARY_SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(scope)==402 and all(sha(root/name)==digest for name,digest in scope.items())
protected={}
for name in ('INITIAL_BOUNDARY_INPUT_SEAL.json','FIXED_M0_SCALE_INPUT_SEAL.json'):
    protected.update(json.loads((root/'refine-logs/incremental_role_objective_v1'/name).read_text())['artifact_sha256'])
assert all(sha(Path(name))==digest for name,digest in protected.items())
family='independent_role_heads_v1_20261006_873'
state=json.loads((root/'logs'/family/'campaign.json').read_text())
assert state['status']=='COMPLETE' and state['report_exit_code']==0
diagnoses=[root/'results/independent_role_heads_fixed_best_v1_20261007_875/RGBNT201_semantic/DIAGNOSIS.json',
    root/'results/independent_role_heads_fixed_best_v1_20261007_875/MSVR310_semantic/DIAGNOSIS.json',
    root/'results/independent_role_heads_fixed_best_pending100_20261007_876/RGBNT100_semantic/DIAGNOSIS.json']
assert all(json.loads(path.read_text())['status']=='COMPLETE' for path in diagnoses)
rows=[]
for dataset in ('RGBNT201','MSVR310','RGBNT100'):
    folder=root/'trained-model'/f'{family}_full_semantic_{dataset}'
    training=json.loads((folder/'training.json').read_text());official=json.loads((folder/'official_metrics.json').read_text())
    assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and official['status']=='COMPLETE'
    assert [row['epoch'] for row in training['history']]==list(range(1,51)) and training['best_epoch']==official['selected_epoch']
    weight=folder/'best_map.pth'
    assert weight.resolve().is_relative_to((root/'trained-model').resolve()) and weight.stat().st_nlink==1
    assert str(weight) not in protected and str(weight.relative_to(root)) not in scope
    assert sha(weight)==official['checkpoint_sha256']
    assert sha(folder/'official_distances.pt')==official['distance_sha256']
    assert sha(folder/'best_epoch_distances.pt')==official['training_best_distance_sha256']
    rows.append(dict(dataset=dataset,path=str(weight),bytes=weight.stat().st_size,sha256=sha(weight),
        best_epoch=official['selected_epoch'],metrics=official['metrics'],
        retained_artifact_sha256={str(path):sha(path) for path in folder.iterdir() if path.is_file() and path!=weight},
        reason='Closed independent-head route failed all six advancement pairs; all three fixed g/c/f consumers complete. No current control/diagnostic/model consumer; retain exact complete training/strict receipts and best/official arrays.'))
journal=root/'logs/closed_independent_head_weight_retirement_20261007_882';assert not journal.exists();journal.mkdir()
qualified=dict(status='QUALIFIED_BEFORE_RETIREMENT',at=datetime.now().astimezone().isoformat(),rows=rows,
    free_bytes_before=shutil.disk_usage(root).free,physical_gpus_queried=[],scope='Three explicit OWN closed non-advancing formal weights only. No full eligibility or M0 gate revision. Current48/fixed9/source402 protected.')
(journal/'QUALIFIED.json').write_text(json.dumps(qualified,indent=2)+'\n')
for row in rows:Path(row['path']).unlink()
assert all(not Path(row['path']).exists() for row in rows)
assert all(sha(Path(name))==digest for name,digest in protected.items())
assert all(sha(root/name)==digest for name,digest in scope.items())
assert all(sha(Path(name))==digest for row in rows for name,digest in row['retained_artifact_sha256'].items())
result=dict(status='THREE_CLOSED_HEAD_WEIGHTS_RETIRED',at=datetime.now().astimezone().isoformat(),rows=rows,
    count=3,bytes=sum(row['bytes'] for row in rows),free_bytes_before=qualified['free_bytes_before'],
    free_bytes_after=shutil.disk_usage(root).free,current_inputs_and_sources_unchanged=True,
    boundary='Authorized timely OWN useless-weight cleanup after all consumers closed. Historical independent-head checkpoint replay now requires retraining; best/official distances, trajectories, strict receipts and fixed g/c/f evidence retained. No authors/current controls/winners removed. Original six M0 FAIL and zero formal increment runs unchanged.')
(journal/'RETIRED.json').write_text(json.dumps(result,indent=2)+'\n')
result['files']={str(path.relative_to(root)):dict(text=path.read_text(),sha256=sha(path)) for path in journal.iterdir() if path.is_file()}
print(json.dumps(result))
