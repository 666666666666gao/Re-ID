"""Executor independent formula replay of all stored source coverage rows."""
from pathlib import Path
from datetime import datetime
import hashlib,json,time
import numpy as np

root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590')
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
summary=json.loads((root/'summary.json').read_bytes())
coverage=json.loads((root/'coverage.json').read_bytes())
assert summary['status']=='COMPLETE_SOURCE_EXTRACTION' and coverage['status']=='COMPLETE_SOURCE_COVERAGE'
protocol=json.loads((repo/'protocols/msvr310_train_oof_v1.json').read_bytes())
q1=json.loads(Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4/q1/summary.json').read_bytes())
result=dict(status='RUNNING',started_at=datetime.now().astimezone().isoformat(),
    model_forwards=0,optimizer_updates=0,conditions=[],full_rows=0,candidate_rows=0,max_ap_error=0.)
destination=root/'coverage_verification.json'
assert not destination.exists()

def independent(values,labels,environments,anchor,pool,cross):
    remaining=[k for k in pool if k!=anchor and not(cross and labels[k]==labels[anchor] and environments[k]==environments[anchor])]
    ranked=sorted(remaining,key=lambda k:(values[k],k))
    hits=np.asarray([labels[k]==labels[anchor] for k in ranked],dtype=bool)
    positive=int(hits.sum());negative=len(hits)-positive
    if not positive:return dict(eligible=False,positives=0,negatives=negative,ap=None,rank1=None,inverted_positives=None)
    ranks=np.arange(1,len(hits)+1);precisions=np.cumsum(hits)[hits]/ranks[hits]
    nearest=min([values[k] for k in remaining if labels[k]!=labels[anchor]],default=np.inf)
    return dict(eligible=True,positives=positive,negatives=negative,ap=float(precisions.mean()),
                rank1=int(hits[0]),inverted_positives=sum(values[k]>nearest for k in remaining if labels[k]==labels[anchor]))

def compare(a,b):
    assert set(a)==set(b)
    for k,v in a.items():
        if k=='ap' and v is not None:
            error=abs(v-b[k]);result['max_ap_error']=max(result['max_ap_error'],error);assert error<1e-12
        else:assert v==b[k],(k,v,b[k])

for cond in summary['conditions']:
    started=time.perf_counter();directory=root/cond['directory'];indices=cond['record_indices']
    assert indices==protocol['folds'][cond['fold']]['source_record_indices'] and indices==sorted(indices)
    labels=[protocol['records'][i]['identity'] for i in indices];scenes=[protocol['records'][i]['scene'] for i in indices]
    assert set(labels)==set(protocol['folds'][cond['fold']]['source_ids'])
    lookup={v:i for i,v in enumerate(indices)}
    logs=Path(q1['folds'][cond['fold']]['endpoints'][cond['endpoint']]['checkpoint']).parent/'memory_steps.jsonl'
    batches=[json.loads(line) for line in logs.read_text().splitlines()]
    pools=[sorted({lookup[i] for i in batch['record_indices']}|{lookup[r['record_index']] for r in batch['memory']}) for batch in batches]
    checked=0
    for path in sorted(directory.glob('*.npy')):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cond['files'][path.name]['sha256']
        x=np.load(path,allow_pickle=False).astype(np.float64)
        # Direct differences from each anchor, independently of the dot-product formula.
        d=np.stack([np.sum((x-x[i])**2,axis=1) for i in range(len(x))])
        # Preserve registered rounding/tie geometry: additionally verify dot-product distance error.
        actual=np.maximum(np.sum(x*x,1)[:,None]+np.sum(x*x,1)[None,:]-2*x@x.T,0)
        assert np.max(np.abs(d-actual))<1e-12
        for cross in (False,True):
            name=path.stem+('_cross_scene' if cross else '_all_identity')
            full=json.loads((directory/(name+'_full.json')).read_bytes())
            assert len(full)==len(indices)
            for i,record in enumerate(full):
                assert record['record_index']==indices[i]
                compare(independent(actual[i],labels,scenes,i,range(len(indices)),cross),{k:v for k,v in record.items() if k!='record_index'})
                result['full_rows']+=1
            with (directory/(name+'_candidates.jsonl')).open() as stream:
                lines=0;cache={};paired=[];missing=0
                for line in stream:
                    row=json.loads(line);step=lines//64;position=lines%64;batch=batches[step]
                    assert row['step']==step+1 and row['anchor_position']==position
                    assert row['record_index']==batch['record_indices'][position]
                    i=lookup[row['record_index']];key=(step,i)
                    if key not in cache:cache[key]=independent(actual[i],labels,scenes,i,pools[step],cross)
                    compare(cache[key],row['candidate'])
                    compare({k:v for k,v in full[i].items() if k!='record_index'},row['full'])
                    if row['candidate']['eligible']:paired.append((row['candidate']['ap'],row['full']['ap']))
                    elif row['full']['eligible']:missing+=1
                    lines+=1;result['candidate_rows']+=1
                assert lines==16640
            saved=next(m for c in coverage['conditions'] if c['directory']==cond['directory']
                       for m in c['metrics'] if m['output']==path.stem and m['cross_scene']==cross)
            legal=[r for r in full if r['eligible']];paired=np.asarray(paired)
            assert saved['full_eligible']==len(legal) and saved['anchor_exposures']==lines
            assert saved['common_eligible']==len(paired) and saved['pool_missing_positive']==missing
            for key,value in dict(full_source_map=100*np.mean([r['ap'] for r in legal]),
                                  common_candidate_map=100*paired[:,0].mean(),common_full_map=100*paired[:,1].mean()).items():
                assert abs(saved[key]-value)<1e-10
            checked+=1
    assert checked==10
    result['conditions'].append(dict(directory=cond['directory'],checked=checked,elapsed_seconds=time.perf_counter()-started))
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['conditions'][-1]),flush=True)
assert len(result['conditions'])==12 and result['candidate_rows']==1996800 and result['full_rows']==82560
result.update(status='PASS_ALL_SOURCE_COVERAGE_ROWS',completed_at=datetime.now().astimezone().isoformat())
destination.write_text(json.dumps(result,indent=2)+'\n')
