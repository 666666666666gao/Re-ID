"""Independent explicit pairwise audit of the saved-source positive diagnostic."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,hashlib,datetime,time,collections
import numpy as np
RUN=pathlib.Path('/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739');R=pathlib.Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
def j(p):return json.loads(p.read_text())
def h(p):
    d=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):d.update(b)
    return d.hexdigest()
start=time.monotonic();cp=RUN/'q1_cpu.json';pp=R/'protocols/msvr310_train_oof_v1.json'
assert h(cp)=='b19622644d45007560e9f930ea52dffea921044f456f8f82d4fc9935d5d55726'
assert h(pp)=='4ff4c60bca3d019929add5788212c526387d93d535a2c52aa7b1c3acfd387cb4'
cpu=j(cp);protocol=j(pp);inputs={str(cp):h(cp),str(pp):h(pp)};endpoints=[];steps=elements=explicit_anchors=pair_comparisons=0
for fi in range(3):
    source=set(protocol['folds'][fi]['source_record_indices'])
    for e in ('control','role_set'):
        d=RUN/'q1'/f'fold_{fi}_{e}'
        for name in ('training.json','memory_steps.jsonl','memory_distances.f32'):
            p=d/name;actual=h(p);proof=cpu['files'][str(p)];assert actual==proof['sha256'] and p.stat().st_size==proof['bytes'];inputs[str(p)]=actual
        rows=[json.loads(line) for line in (d/'memory_steps.jsonl').read_text().splitlines()];training=j(d/'training.json')
        assert len(rows)==len(training['steps'])==260
        phases={k:collections.Counter() for k in ('all','warmup','post_warmup','last65')};extra=collections.Counter()
        with (d/'memory_distances.f32').open('rb') as f:
            for si,(row,tr) in enumerate(zip(rows,training['steps'])):
                assert row['step']==tr['step']==si+1 and row['record_indices']==tr['sampled_record_indices'] and len(row['record_indices'])==64
                assert f.tell()==row['distance_offset_bytes'] and row['history_anchor_count']==0 and row['current_anchor_count']==64
                rec=row['record_indices']+[m['record_index'] for m in row['memory']];assert set(rec)<=source
                truth=[protocol['records'][k] for k in rec];ids=np.array([r['identity'] for r in truth]);sc=np.array([r['scene'] for r in truth]);records=np.array(rec)
                assert ids.tolist()==row['identities']+[m['identity'] for m in row['memory']]
                assert sc.tolist()==row['scenes']+[m['scene'] for m in row['memory']]
                raw=np.fromfile(f,dtype='<f4',count=4*64*len(rec));assert len(raw)==row['distance_float_count'] and np.isfinite(raw).all() and raw.min()>=0
                distance=raw.reshape(4,64,len(rec))[0];steps+=1;elements+=len(raw)
                count=collections.Counter(steps=1,anchor_exposures=64);ge_wrong=0
                for a in range(64):
                    positive=np.flatnonzero((ids==ids[a])&(np.arange(len(ids))!=a));negative=np.flatnonzero(ids!=ids[a]);assert len(positive)>0 and len(negative)>0
                    pval=distance[a,positive];nval=distance[a,negative];hp=float(max(pval))
                    # Explicit comparisons for EVERY anchor, without searchsorted.
                    matrix=pval[:,None]>nval[None,:];equal=pval[:,None]==nval[None,:]
                    perpositive=matrix.sum(axis=1);pertie=equal.sum(axis=1);affected=matrix.any(axis=1);nonmax=pval<hp;cross=sc[positive]!=sc[a];historical=positive>=64
                    explicit_anchors+=1;pair_comparisons+=matrix.size;ge_wrong+=int(hp>=float(min(nval)))
                    count['anchors_with_any_strict_inversion']+=int(matrix.any())
                    count['anchors_with_cross_scene_positive']+=int(cross.any())
                    count['anchors_with_cross_scene_strict_inversion']+=int(affected[cross].any())
                    count['anchors_with_nonmax_strict_inversion']+=int(affected[nonmax].any())
                    count['anchors_with_nonmax_cross_scene_strict_inversion']+=int(affected[cross&nonmax].any())
                    count['anchors_all_hardest_positives_same_scene_with_cross_available']+=int(cross.any() and not cross[pval==hp].any())
                    extra['anchors_with_multiple_maximum_positive_positions']+=int((pval==hp).sum()>1)
                    extra['affected_anchors_with_multiple_maximum_positive_positions']+=int((pval==hp).sum()>1 and matrix.any())
                    for prefix,mask in [('all_positive',np.ones(len(positive),dtype=bool)),('cross_scene_positive',cross)]:
                        count[prefix+'_positions']+=int(mask.sum());count[prefix+'_distinct_record_exposures']+=len(set(records[positive[mask]].tolist()))
                        count[prefix+'_strict_inversion_pairs']+=int(perpositive[mask].sum());count[prefix+'_negative_distance_tie_pairs']+=int(pertie[mask].sum())
                        count[prefix+'_affected_positions']+=int((affected&mask).sum());count[prefix+'_affected_nonmax_positions']+=int((affected&mask&nonmax).sum())
                        count[prefix+'_affected_max_positions']+=int((affected&mask&~nonmax).sum());count[prefix+'_affected_historical_positions']+=int((affected&mask&historical).sum())
                        count[prefix+'_nonmax_strict_inversion_pairs']+=int(perpositive[mask&nonmax].sum())
                    count['same_record_positive_position_exposures']+=int((records[positive]==records[a]).sum())
                assert ge_wrong==row['statistics']['expanded_wrong_order_anchors']
                assert count['anchors_with_any_strict_inversion']==ge_wrong  # Observed equality only; explicit ties are separately counted.
                for prefix in ('all_positive','cross_scene_positive'):
                    assert count[prefix+'_affected_positions']==count[prefix+'_affected_max_positions']+count[prefix+'_affected_nonmax_positions']
                phases['all'].update(count);phases['warmup' if si<65 else 'post_warmup'].update(count)
                if si>=195:phases['last65'].update(count)
            assert not f.read(1)
        assert phases['all']['steps']==260 and phases['warmup']['steps']==65 and phases['post_warmup']['steps']==195 and phases['last65']['steps']==65
        endpoints.append(dict(fold=fi,endpoint=e,phases={k:dict(v) for k,v in phases.items()},extra_maximum_tie_evidence=dict(extra)))
aggregates={e:{} for e in ('control','role_set')}
for e in aggregates:
    selected=[r for r in endpoints if r['endpoint']==e]
    for phase in ('all','warmup','post_warmup','last65'):
        result=collections.Counter()
        for r in selected:result.update(r['phases'][phase])
        aggregates[e][phase]=dict(result)
assert steps==1560 and elements==116501504 and explicit_anchors==99840
print(json.dumps(dict(status='PASS_INDEPENDENT_ALL_ANCHOR_EXPLICIT_POSITIVE_NEGATIVE_REPLAY',time=datetime.datetime.now().astimezone().isoformat(),inputs=inputs,endpoints=endpoints,aggregates=aggregates,checked_steps=steps,read_four_space_distance_elements=elements,fused_distance_elements_analyzed=elements//4,explicit_anchor_pair_matrices=explicit_anchors,explicit_positive_negative_pair_comparisons=pair_comparisons,model_forwards=0,optimizer_updates=0,checkpoint_loads=0,heldout_rankings_read=0,image_reads=0,remote_writes=0,elapsed_seconds=time.monotonic()-start,scope='Descriptive saved-source snapshots only; explicit pairwise comparisons at every anchor, exact strict inversion/tie/all-positive maximum definitions; no parameter gradient or final model generalization claim.')))
