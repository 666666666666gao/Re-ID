from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
j=root/'logs/independent_role_heads_fixed_best_pending100_launch_20261007_876'
out=root/'results/independent_role_heads_fixed_best_pending100_20261007_876'
qualified=json.loads((j/'QUALIFIED.json').read_text())
assert json.loads((j/'EXIT.json').read_text())['exit_code']==0
for n in ('LAUNCH.json','CHILD.json'):
    assert not (Path('/proc')/str(json.loads((j/n).read_text())['pid'])).exists()
assert all(sha(Path(n))==d for n,d in qualified['original_failure_and_completed_sha256'].items())
assert all(sha(Path(n))==d for n,d in qualified['administrative_contract_sha256'].items())
seal=json.loads((root/'refine-logs/independent_role_heads_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(seal['source_sha256'])==388 and len(seal['artifact_sha256'])==241
assert all(sha(root/n)==d for n,d in seal['source_sha256'].items())
assert all(sha(Path(n))==d for n,d in seal['artifact_sha256'].items())
paths={dataset:root/'results/independent_role_heads_fixed_best_v1_20261007_875'/f'{dataset}_semantic'
    for dataset in ('RGBNT201','MSVR310')}
paths['RGBNT100']=out/'RGBNT100_semantic'
counts={'RGBNT201':(836,836),'MSVR310':(591,1055),'RGBNT100':(1715,8575)}
reports={}
for dataset,folder in paths.items():
    report=json.loads((folder/'DIAGNOSIS.json').read_text())
    row=next(r for r in seal['rows'] if (r['dataset'],r['variant'])==(dataset,'semantic'))
    assert report['status']=='COMPLETE' and report['dataset']==dataset
    assert report['selected_epoch']==row['best_epoch'] and report['input_checkpoint_sha256']==row['checkpoint_sha256']
    assert report['model_state_before_sha256']==report['model_state_after_sha256']
    assert len(report['scores']['fused']['average_precision'])==counts[dataset][0]
    for split,count in zip(('query','gallery'),counts[dataset]):
        assert all(v['count']==count for v in report['diagnostic'][split].values())
    assert set(report['artifacts'])=={'query_statistics.pt','gallery_statistics.pt','diagnostic_distances.pt'}
    for n,v in report['artifacts'].items():
        assert (folder/n).stat().st_size==v['bytes'] and sha(folder/n)==v['sha256']
    reports[dataset]=dict(selected_epoch=report['selected_epoch'],
        metrics={n:v['metrics'] for n,v in report['scores'].items()},
        comparisons={n:{k:v for k,v in p.items() if k not in ('query_changes','identity_changes')}
            for n,p in report['comparisons'].items()},diagnostic=report['diagnostic'],
        readout_gain=report['readout_gain'],elapsed_seconds=report['elapsed_seconds'],
        fused_distance_max_absolute_difference_from_original=report['fused_distance_max_absolute_difference_from_original'],
        artifacts=report['artifacts'])
names=[j/n for n in ('QUALIFIED.json','supervisor.py','LAUNCH.json','CHILD.json','EXIT.json','controller.log','supervisor.log')]
names.append(out/'RGBNT100_semantic/DIAGNOSIS.json')
print(json.dumps(dict(status='THREE_DIAGNOSES_COMPLETE_ORIGINAL_FAILED_PLUS_PENDING100',
    at=datetime.now().astimezone().isoformat(),text_files={str(p.relative_to(root)):sha(p) for p in names},
    source_count=388,input_artifact_count=241,optimizer_updates=0,record_exposures=13608,
    reports=reports,original_failure_and_completed_unchanged=True,
    saved_diagnostic_bytes=sum(v['bytes'] for r in reports.values() for v in r['artifacts'].values()),
    free_bytes=shutil.disk_usage(root).free,
    boundary='Original queue remains FAILED2/3. Only never-forwarded100 resumed independently. All three fixed-best reports and9 output artifacts now physically verified; no model/scoring replay, no source/best selection or updates.')))
