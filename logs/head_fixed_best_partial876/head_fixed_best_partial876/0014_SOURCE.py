from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
out=root/'results/independent_role_heads_fixed_best_v1_20261007_875'
j=root/'logs/independent_role_heads_fixed_best_launch_20261007_875'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
campaign=json.loads((out/'campaign.json').read_text())
assert campaign['status']=='FAILED' and campaign['optimizer_updates']==0
assert len(campaign['jobs'])==3
assert [(r['dataset'],r['status'],r['exit_code']) for r in campaign['jobs']]==[('RGBNT201','COMPLETE',0),('MSVR310','COMPLETE',0),('RGBNT100','FAILED',1)]
assert json.loads((j/'EXIT.json').read_text())['exit_code']==1
launch=json.loads((j/'LAUNCH.json').read_text())
child=json.loads((j/'CHILD.json').read_text())
assert not (Path('/proc')/str(launch['pid'])).exists()
assert not (Path('/proc')/str(child['pid'])).exists()
seal_path=root/'refine-logs/independent_role_heads_fixed_best_diagnosis_v1/INPUT_SEAL.json'
seal=json.loads(seal_path.read_text())
assert len(seal['source_sha256'])==388 and len(seal['artifact_sha256'])==241
assert all(sha(root/n)==d for n,d in seal['source_sha256'].items())
assert all(sha(Path(n))==d for n,d in seal['artifact_sha256'].items())
names=[j/n for n in ('QUALIFIED.json','supervisor.py','LAUNCH.json','CHILD.json',
    'EXIT.json','controller.log','supervisor.log')]+[out/'campaign.json']
reports={}
names.append(out/'RGBNT100_semantic.log')
assert not (out/'RGBNT100_semantic').exists()
assert 'assert shutil.disk_usage(ROOT).free>=STORAGE_BYTES' in (out/'RGBNT100_semantic.log').read_text()
expected_counts={'RGBNT201':(836,836),'MSVR310':(591,1055),'RGBNT100':(1715,8575)}
for dataset in ('RGBNT201','MSVR310'):
    folder=out/f'{dataset}_semantic'
    p=folder/'DIAGNOSIS.json';report=json.loads(p.read_text())
    row=next(r for r in seal['rows'] if (r['dataset'],r['variant'])==(dataset,'semantic'))
    assert report['status']=='COMPLETE' and report['dataset']==dataset
    assert report['selected_epoch']==row['best_epoch']
    assert report['input_checkpoint_sha256']==row['checkpoint_sha256']
    assert report['model_state_before_sha256']==report['model_state_after_sha256']
    assert len(report['scores'])==4
    assert len(report['scores']['fused']['average_precision'])==expected_counts[dataset][0]
    for split,count in zip(('query','gallery'),expected_counts[dataset]):
        assert all(v['count']==count for v in report['diagnostic'][split].values())
    assert set(report['artifacts'])=={'query_statistics.pt','gallery_statistics.pt','diagnostic_distances.pt'}
    assert not (folder/'query_features.pt').exists() and not (folder/'gallery_features.pt').exists()
    for n,value in report['artifacts'].items():
        assert (folder/n).stat().st_size==value['bytes'] and sha(folder/n)==value['sha256']
    names.extend([p,out/f'{dataset}_semantic.log'])
    reports[dataset]=dict(selected_epoch=report['selected_epoch'],
        metrics={n:v['metrics'] for n,v in report['scores'].items()},
        comparisons={n:{k:v for k,v in r.items() if k not in ('query_changes','identity_changes')}
            for n,r in report['comparisons'].items()},
        diagnostic=report['diagnostic'],readout_gain=report['readout_gain'],
        elapsed_seconds=report['elapsed_seconds'],
        fused_distance_max_absolute_difference_from_original=report['fused_distance_max_absolute_difference_from_original'],
        artifacts=report['artifacts'])
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),
    status='TWO_COMPLETED_DIAGNOSES_RGBNT100_NOT_STARTED_STORAGE_FAILURE',source_count=388,input_artifact_count=241,
    seal_sha256=sha(seal_path),optimizer_updates=0,record_exposures=3318,
    text_files={str(p.relative_to(root)):sha(p) for p in names},reports=reports,
    saved_diagnostic_bytes=sum(v['bytes'] for r in reports.values() for v in r['artifacts'].values()),
    free_bytes=shutil.disk_usage(root).free,
    boundary='Completed-only read-only intake and physical SHA verification. No NN/scorer/report replay, no checkpoint selection, updates, deletion or original result mutation. Same-model global/correction analysis is descriptive and official-best selected, not independent training ablation or causal proof.')))
