"""Replay registered source labels and queue only; no scores, images or training."""
from pathlib import Path
from collections import Counter,OrderedDict,defaultdict
import hashlib,json,sys

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
output=Path(sys.argv[1]);assert not output.exists()
bindings={
 'protocols/msvr310_train_oof_v1.json':'4ff4c60bca3d019929add5788212c526387d93d535a2c52aa7b1c3acfd387cb4',
 'evidence/msvr310_style_t0_runtime_binding_20260907/msvr_style_metadata_remote_numpy_20260907.json':'87dd4773b75e6485c545149c5e053a1b4c930b9f8babdc5a49ce1d8165c3d0d4',
 'configs/MSVR310/TriFusion-smooth-ap-paired-v1.json':'974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302'}
data=[]
for path,sha in bindings.items():
    raw=(repo/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha;data.append(json.loads(raw))
protocol,metadata,config=data
t0_path=Path('C:/Users/gb/.codex_tmp/smooth_ap_m0_complete_20260908/t0.json')
t0=json.loads(t0_path.read_bytes())
records=protocol['records'];assert len(records)==1032
result=[]
for fold,md,queue_ref in zip(protocol['folds'],metadata['folds'],t0['folds'],strict=True):
    source=set(fold['source_record_indices']);envs=defaultdict(set)
    for idx in source:envs[records[idx]['identity']].add(records[idx]['scene'])
    eligible_ids={identity for identity,scenes in envs.items() if len(scenes)>1}
    assert len(eligible_ids)==40
    cache=OrderedDict();batches=[];phases=defaultdict(Counter);identity_counts=defaultdict(Counter)
    assert len(md['batches'])==len(queue_ref['steps'])==260
    for step,(batch,qref) in enumerate(zip(md['batches'],queue_ref['steps'],strict=True)):
        indices=batch['record_indices'];assert len(indices)==64 and set(indices)<=source
        cache=OrderedDict((idx,stored) for idx,stored in cache.items() if step-stored<=config['memory']['maximum_age'])
        history=[idx for idx in cache if idx not in indices]
        assert qref==dict(step=step+1,historical_records=len(history),ages=[step-cache[idx] for idx in history])
        candidates=indices+history
        counts=Counter();cross_hist=Counter();legal_anchor_ids=set()
        for i,idx in enumerate(indices):
            q=records[idx];identity=q['identity'];scene=q['scene']
            positive=[j for j,k in enumerate(candidates) if j!=i and records[k]['identity']==identity]
            cross=[j for j in positive if records[candidates[j]]['scene']!=scene]
            negative=[j for j,k in enumerate(candidates) if records[k]['identity']!=identity]
            assert positive and negative
            current_cross=[j for j in cross if j<64]
            globally_eligible=identity in eligible_ids
            assert globally_eligible or not cross
            if cross:legal_anchor_ids.add(identity)
            row=dict(anchor_exposures=1,all_positive_positions=len(positive),
                cross_positive_positions=len(cross),same_scene_positive_positions=len(positive)-len(cross),
                cross_unique_records=len({candidates[j] for j in cross}),negative_positions=len(negative),
                anchors_with_current_cross=int(bool(current_cross)),anchors_with_pool_cross=int(bool(cross)),
                anchors_rescued_by_history=int(bool(cross) and not current_cross),
                globally_eligible_anchor_exposures=int(globally_eligible),
                globally_eligible_but_pool_missing=int(globally_eligible and not cross),
                globally_ineligible_anchor_exposures=int(not globally_eligible))
            counts.update(row);identity_counts[identity].update(row);cross_hist[len(cross)]+=1
        assert counts['anchor_exposures']==64
        assert counts['anchors_with_pool_cross']+counts['globally_eligible_but_pool_missing']+counts['globally_ineligible_anchor_exposures']==64
        counts.update(batches=1,batches_without_legal_anchor=int(not legal_anchor_ids))
        names=['all','post_warmup' if step>=65 else 'warmup']
        if step>=195:names.append('last65')
        for name in names:phases[name].update(counts)
        batches.append(dict(step=step+1,history_records=len(history),counts=dict(counts),cross_positive_count_histogram=dict(cross_hist),eligible_identities=len(legal_anchor_ids)))
        if step>=config['memory']['warmup_steps']:
            for idx in indices:cache.pop(idx,None);cache[idx]=step
            while len(cache)>config['memory']['capacity']:cache.popitem(last=False)
    result.append(dict(fold=fold['fold'],source_ids=len(envs),cross_scene_source_ids=len(eligible_ids),source_records=len(source),phases={k:dict(v) for k,v in phases.items()},identities=[dict(identity=i,available_source_scenes=sorted(envs[i]),counts=dict(c)) for i,c in sorted(identity_counts.items())],batches=batches))
assert sum(f['phases']['all']['batches'] for f in result)==780
out=dict(status='COMPLETE_REGISTERED_SOURCE_QUEUE_CROSS_SCENE_SUPPORT_REPLAY',input_bindings=bindings,t0_sha256=hashlib.sha256(t0_path.read_bytes()).hexdigest(),folds=result,scope='One shared registered 780-batch sequence, 49920 repeated anchor exposures, not six separately executed endpoints or independent instances. No current Q1 files or scores read. Source labels plus exact T0-validated FIFO age/capacity/warmup replay. Counts quantify candidate support only, not gradient mass, difficulty, retrieval benefit or novel method. Same-identity same-scene candidates would be excluded from the proposed ranking term, never relabelled negative. Ineligible anchors and identities remain in other supervision and candidate negatives. No new experiment, loss normalization or training mask registered.')
output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=out['status'],folds=[dict(fold=f['fold'],post_warmup=f['phases']['post_warmup']) for f in result],bytes=output.stat().st_size)))
