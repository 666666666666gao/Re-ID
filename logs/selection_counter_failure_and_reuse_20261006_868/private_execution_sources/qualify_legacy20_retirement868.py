from pathlib import Path
import hashlib, json

root=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selection_legacy20_metadata868')
meta=json.loads((root/'REMOTE.json').read_bytes())
remote=Path('/data/gaob/Re-ID/Trifusion')
rows=[]
for row in meta['rows']:
    p=Path(row['path']); t=row['training']
    receipt_path=root/'received'/p.parent.relative_to(remote)/'official_metrics.json'
    assert receipt_path.is_file(),p
    receipt=json.loads(receipt_path.read_bytes())
    assert t['status']=='FIXED_EPOCH20_TRAINING_COMPLETE' and t['training']['epochs']==20
    assert receipt['status']=='COMPLETE' and receipt['fixed_epoch']==20
    for key in ('dataset','method','seed','protocol_sha256'):
        assert t[key]==receipt[key],(p,key)
    assert t['checkpoint_sha256']==receipt['role_checkpoint_sha256']
    init=t['initializer']
    assert init['author_checkpoint_sha256']==receipt['author_checkpoint_sha256']
    assert receipt['independent_upstream_metrics_equal']
    group=(t['schema'],t['dataset'],t['method'],t['protocol_sha256'],init['author_checkpoint_sha256'],init['signal_state_sha256'],init.get('baseline_kind'))
    rows.append(dict(path=row['path'],bytes=row['bytes'],checkpoint_sha256=t['checkpoint_sha256'],group=group,
        training_sha256=hashlib.sha256((root/'received'/p.parent.relative_to(remote)/'training.json').read_bytes()).hexdigest(),
        receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),metrics=receipt['outputs']['fused']['metrics'],
        distance_arrays=receipt['distance_arrays'],distance_arrays_sha256=receipt['distance_arrays_sha256']))
def dominates(a,b):
    return all(a['metrics'][k]>=b['metrics'][k] for k in ('mAP','Rank-1','Rank-5','Rank-10')) and a['metrics']['mAP']>b['metrics']['mAP']
targets=[];retained=[]
for row in rows:
    peers=[r for r in rows if r['group']==row['group']]
    winners=[r for r in peers if not any(dominates(x,r) for x in peers)]
    candidates=[r for r in winners if dominates(r,row)]
    if candidates:
        winner=max(candidates,key=lambda r:r['metrics']['mAP'])
        targets.append(dict(target=row,winner=winner))
    else:retained.append(row)
out=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selection_legacy20_qualification868')
assert not out.exists();out.mkdir()
result=dict(status='LOCAL_METADATA_QUALIFIED_NOT_DELETED',targets=targets,retained=retained,
    prospective_retired_bytes=sum(r['target']['bytes'] for r in targets),
    boundary='Same dataset/method/protocol/author+Signal state. Every retained dominating winner has mAP higher and allCMC no lower. Own completed fixed20 weights only. PhysicalSHA/current dependency qualification still required; no deletion yet. All official receipts and distance arrays retained.')
(out/'PLAN.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(targets=len(targets),prospective_retired_bytes=result['prospective_retired_bytes'],pairs=[dict(target=r['target']['path'],winner=r['winner']['path']) for r in targets]),indent=2))
