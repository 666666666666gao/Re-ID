import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path
from datetime import datetime, timezone

root = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
run = Path('/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d')
paths = dict(protocol=root/'protocols/msvr310_train_oof_v1.json',
             labels=root/'evidence/vehicle_query_protocol_labels_20260905.json',
             support=root/'evidence/smooth_ap_cross_scene_support_20260909/smooth_ap_cross_scene_support_20260909.json',
             sequence=root/'evidence/msvr310_style_t0_runtime_binding_20260907/msvr_style_metadata_remote_numpy_20260907.json',
             t0=run/'t0.json',
             loader=Path('/root/autodl-tmp/trifusion-v2/comparators/Signal-cd1b0a6/data/datasets/msvr310.py'))
data = {k:json.loads(p.read_bytes()) for k,p in paths.items() if k!='loader'}
protocol, labels, support, sequence, t0 = (data[k] for k in ('protocol','labels','support','sequence','t0'))
hashes = {k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()}
assert hashes['labels']['sha256']==protocol['label_evidence_sha256']
train = next(x for x in labels['datasets'] if x['dataset']=='MSVR310')['record_manifest']['bounding_box_train']
records = protocol['records']
assert len(train)==len(records)==1032
ids = sorted({r['identity'] for r in records})
scenes = {i:sorted({r['scene'] for r in records if r['identity']==i}) for i in ids}
heldouts = [set() for _ in range(3)]
for group in ([i for i in ids if len(scenes[i])>1], [i for i in ids if len(scenes[i])==1]):
    for position,i in enumerate(group):
        heldouts[position%3].add(i)
for i,(src,r) in enumerate(zip(train,records)):
    assert r['index']==i
    assert all(src[k]==r[k] for k in ('identity','camera','scene'))
    name = Path(src['path']).name
    assert (r['identity'],r['camera'],r['scene'])==(int(name[:4]),int(name[11]),int(name[6:9]))
    expected = [str(Path('bounding_box_train')/Path(src['path']).parts[0]/m/name) for m in ('vis','ni','th')]
    assert r['paths']==expected

fold_checks=[]
support_checks=[]
zero_batches=[]
for f,md,sup,trow in zip(protocol['folds'],sequence['folds'],support['folds'],t0['folds'],strict=True):
    fi=f['fold']; assert fi==md['fold']==sup['fold']==trow['fold']
    assert set(f['heldout_ids'])==heldouts[fi]
    assert set(f['source_ids'])==set(ids)-heldouts[fi]
    assert f['source_record_indices']==[r['index'] for r in records if r['identity'] in f['source_ids']]
    assert f['gallery_record_indices']==[r['index'] for r in records if r['identity'] in f['heldout_ids']]
    assert f['source_label_map']=={str(i):j for j,i in enumerate(f['source_ids'])}
    assert len(set(f['source_record_indices']) & set(f['gallery_record_indices']))==0
    gallery=[records[i] for i in f['gallery_record_indices']]
    expected_queries=[]
    for position,r in enumerate(gallery):
        positive=sum(x['identity']==r['identity'] and x['scene']!=r['scene'] for x in gallery)
        if positive:
            expected_queries.append((r['index'],position,positive))
    assert expected_queries==[(x['record_index'],x['gallery_position'],x['valid_positives']) for x in f['query_rows']]
    fold_checks.append(dict(fold=fi,source_identities=len(f['source_ids']),heldout_identities=len(f['heldout_ids']),
                            source_records=len(f['source_record_indices']),heldout_records=len(gallery),valid_queries=len(expected_queries)))
    cache=OrderedDict(); phase={k:Counter() for k in ('all','warmup','post_warmup','last65')}; identity={i:Counter() for i in f['source_ids']}
    assert len(md['batches'])==len(sup['batches'])==len(trow['steps'])==260
    for step,(batch,saved,tsaved) in enumerate(zip(md['batches'],sup['batches'],trow['steps'],strict=True)):
        current=batch['record_indices']; assert len(current)==64 and set(current)<=set(f['source_record_indices'])
        assert sorted(Counter(records[i]['identity'] for i in current).values())==[8]*8
        for i in list(cache):
            if step-cache[i]>8: del cache[i]
        historical=[i for i in cache if i not in set(current)]
        candidates=current+historical; counts=Counter(); cross_counts=[]; eligible_ids=set()
        for position,index in enumerate(current):
            q=records[index]
            positive=[j for j,i in enumerate(candidates) if records[i]['identity']==q['identity'] and j!=position]
            cross=[j for j in positive if records[candidates[j]]['scene']!=q['scene']]
            current_cross=[j for j in cross if j<64]
            eligible=len(scenes[q['identity']])>1
            c=dict(anchor_exposures=1,all_positive_positions=len(positive),cross_positive_positions=len(cross),
                   same_scene_positive_positions=len(positive)-len(cross),cross_unique_records=len({candidates[j] for j in cross}),
                   negative_positions=sum(records[i]['identity']!=q['identity'] for i in candidates),
                   anchors_with_current_cross=int(bool(current_cross)),anchors_with_pool_cross=int(bool(cross)),
                   anchors_rescued_by_history=int(bool(cross) and not current_cross),globally_eligible_anchor_exposures=int(eligible),
                   globally_eligible_but_pool_missing=int(eligible and not cross),globally_ineligible_anchor_exposures=int(not eligible))
            counts.update(c); identity[q['identity']].update(c); cross_counts.append(len(cross))
            if cross: eligible_ids.add(q['identity'])
        counts['batches']=1; counts['batches_without_legal_anchor']=int(not any(cross_counts))
        assert dict(counts)==saved['counts']
        assert saved['history_records']==len(historical) and saved['step']==step+1
        assert saved['eligible_identities']==len(eligible_ids)
        assert saved['cross_positive_count_histogram']==dict(Counter(str(n) for n in cross_counts))
        assert tsaved['step']==step+1 and tsaved['historical_records']==len(historical)
        assert tsaved['cross_scene_positive_counts']==cross_counts
        assert tsaved['eligible_anchors']==sum(n>0 for n in cross_counts)
        assert tsaved['ages']==[step-cache[i] for i in historical]
        for key in ('all', 'warmup' if step<65 else 'post_warmup'):
            phase[key].update(counts)
        if step>=195: phase['last65'].update(counts)
        if not any(cross_counts): zero_batches.append(dict(fold=fi,step=step+1,post_warmup=step>=65))
        if step>=65:
            for i in current: cache.pop(i,None); cache[i]=step
            while len(cache)>512:cache.popitem(last=False)
    assert {k:dict(v) for k,v in phase.items()}==sup['phases']
    for saved in sup['identities']:
        assert saved['available_source_scenes']==scenes[saved['identity']]
        assert saved['counts']==dict(identity[saved['identity']])
    support_checks.append(dict(fold=fi,batches=260,post_warmup_eligible=phase['post_warmup']['anchors_with_pool_cross'],
                               post_warmup_anchors=phase['post_warmup']['anchor_exposures'],all_counts_exact=True))
result=dict(status='PASS_STATIC_PROVENANCE_AND_REGISTERED_TEXT_REPLAY',collected_at=datetime.now(timezone.utc).isoformat(),
            hashes=hashes,records_checked=1032,identities_checked=len(ids),folds=fold_checks,
            support_batches_checked=780,anchor_exposures_checked=49920,support=support_checks,
            zero_eligible_batches=zero_batches,official_image_reads=0,model_forwards=0,optimizer_updates=0,
            loader_text=paths['loader'].read_text())
print(json.dumps(result,ensure_ascii=False))
