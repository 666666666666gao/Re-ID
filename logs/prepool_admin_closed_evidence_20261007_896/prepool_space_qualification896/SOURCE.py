from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and (proc/'cmdline').is_file():
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in (proc/'cmdline').read_bytes().split(bytes([0])))
scope=json.loads((root/'refine-logs/prepool_dense_correspondence_v1/SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(scope)==428 and all(sha(root/n)==d for n,d in scope.items())
protected={}
for name in ('INPUT_SEAL.json','INITIAL_BOUNDARY_INPUT_SEAL.json'):
    protected.update(json.loads((root/'refine-logs/incremental_role_objective_v1'/name).read_text())['artifact_sha256'])
assert len(protected)==48 and all(sha(Path(n))==d for n,d in protected.items())
campaign=root/'logs/native_research_v6_20261003_794/campaign.json'
assert json.loads(campaign.read_text())['status']=='COMPLETE' and json.loads(campaign.read_text())['report_exit_code']==0
consumer=root/'results/native_fixed_best_20261004_811'
assert json.loads((consumer/'campaign.json').read_text())['status']=='COMPLETE'
specs=[('native','RGBNT201','c35b2ea4ee722cb932623e8bce1f688a797f6b29fa46c96ad6c73f18e08e0a01'),
('native','MSVR310','306c24b138f20fcd4fdb00348e8c2b34e096fd8b1d5f05c9a9bba143de9c5f5f'),
('native','RGBNT100','448d25bb1ec159e208011e4287b588b44de620a5c17c69abefd4d56212327ecd'),
('semantic','RGBNT201','084b96bf963f02f1ae646c7df98763b914c64b6ef60d5ea12ab3d77b1d38b0ca'),
('semantic','RGBNT100','34c048f0ad92229e3b629b2f34f2125431fbddc13f071e8a4331414594456459')]
rows=[];retained={str(campaign):sha(campaign),str(consumer/'campaign.json'):sha(consumer/'campaign.json')}
for variant,dataset,digest in specs:
    folder=root/f'trained-model/native_research_v6_20261003_794_full_{variant}_{dataset}'
    training=json.loads((folder/'training.json').read_text());official=json.loads((folder/'official_metrics.json').read_text())
    assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and official['status']=='COMPLETE'
    assert [r['epoch'] for r in training['history']]==list(range(1,51))
    assert training['best_epoch']==official['selected_epoch']
    assert official['independent_upstream_metrics_equal'] and not official['reranking']
    best=folder/'best_map.pth'
    assert best.resolve().is_relative_to((root/'trained-model').resolve()) and best.stat().st_nlink==1
    assert str(best) not in protected and str(best.relative_to(root)) not in scope
    assert sha(best)==digest==official['checkpoint_sha256']
    assert sha(folder/'official_distances.pt')==official['distance_sha256']
    assert sha(folder/'best_epoch_distances.pt')==official['training_best_distance_sha256']
    diagnosis=consumer/f'{dataset}_{variant}/DIAGNOSIS.json';d=json.loads(diagnosis.read_text())
    assert d['status']=='COMPLETE' and d['input_checkpoint_sha256']==digest
    assert d['selected_epoch']==official['selected_epoch']
    assert d['model_state_before_sha256']==d['model_state_after_sha256']
    for p in list(folder.iterdir())+list(diagnosis.parent.rglob('*')):
        if p.is_file() and p!=best:retained[str(p)]=sha(p)
    rows.append(dict(path=str(best),bytes=best.stat().st_size,sha256=digest,dataset=dataset,variant=variant,
        selected_epoch=official['selected_epoch'],metrics=official['metrics'],diagnosis_sha256=sha(diagnosis),
        reason='Closed old794 inferior to current independently trained global controls; known fixed-best811 consumer complete. No current pilot dependency.'))
assert len(rows)==5 and sum(r['bytes'] for r in rows)==1789083787
for name in ('logs/prepool_dense_launch_20261007_895','logs/prepool_dense_v1_20261007_895','results/prepool_dense_complete_20261007_895'):
    assert not (root/name).exists()
value=dict(status='FIVE_CLOSED_OLD794_OWN_WEIGHTS_QUALIFIED_NOT_RETIRED',at=datetime.now().astimezone().isoformat(),rows=rows,
    count=5,bytes=sum(r['bytes'] for r in rows),free_bytes=shutil.disk_usage(root).free,required_new_pilot_bytes=3758096384,
    protected_artifact_sha256=protected,retained_artifact_sha256=retained,source_count=428,
    boundary='Read-only five exact closed OWN old794 weights; no GPU/NN/deletion. Current48 dependencies and428sources verified. OldMSVRsemantic/strongwinners/public authors remain. Histories/strict metrics/distances/diagnostic evidence retained; retiring PTH requires regeneration for direct model replay.')
print(json.dumps(value))
