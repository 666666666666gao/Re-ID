"""Independent CPU arithmetic from saved M0 arrays and dataset filenames only.

Does not import the experiment's verifier, objective, sampler or model code.
Does not open any image, instantiate a model, or consume a Q1 artifact.
"""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np

START=time.time()
ROOT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739')
EXPERTS=('cnn','transformer','mamba')
def read(path):return json.loads(Path(path).read_bytes())
spec=read(ROOT/'configs/MSVR310/TriFusion-role-set-paired-v1.json')
config=read(ROOT/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
base=read(ROOT/config['BASELINE']['CONFIG'])
protocol=read(ROOT/base['protocol'])
meta=read(ROOT/config['SOURCE_METADATA']['PATH'])
summary=read(RUN/'m0/summary.json')
cpu=read(RUN/'m0_cpu.json')
t0=read(RUN/'t0.json')
records=protocol['records'];record_ids=set(range(len(records)))
assert len(records)==1032 and [r['index'] for r in records]==list(range(1032))
assert all(r['paths'][0].startswith('bounding_box_train/') for r in records)
assert len({Path(r['paths'][0]).name for r in records})==1032
observed_paths={p.relative_to(base['dataset_root']).as_posix() for p in (Path(base['dataset_root'])/'bounding_box_train').rglob('*.jpg')}
wanted_paths={p for r in records for p in r['paths']}
assert observed_paths==wanted_paths and len(wanted_paths)==3096
for r in records:
    names=[Path(p).name for p in r['paths']]
    assert len(set(names))==1
    assert (int(names[0][:4]),int(names[0][11]),int(names[0][6:9]))==(r['identity'],r['camera'],r['scene'])
    assert [Path(p).parts[2] for p in r['paths']]==['vis','ni','th']
    assert int(Path(r['paths'][0]).parts[1])==r['identity']
identities=sorted({r['identity'] for r in records})
scene_sets={i:{r['scene'] for r in records if r['identity']==i} for i in identities}
eligible=[i for i in identities if len(scene_sets[i])>1]
single=[i for i in identities if len(scene_sets[i])==1]
assert (len(identities),len(eligible),len(single))==(155,60,95)
fold_results=[]
sampler_batches={}

def sampled_batches(fold):
    rng=random.Random(42);nrng=np.random.RandomState(42)
    groups={}
    for i in fold['source_record_indices']:
        groups.setdefault(records[i]['identity'],[]).append(i)
    batches=[]
    for epoch in range(20):
        queues={}
        for identity,source in groups.items():
            values=list(source)
            if len(values)<8:values=nrng.choice(values,size=8,replace=True)
            rng.shuffle(values)
            queues[identity]=[list(map(int,values[i:i+8])) for i in range(0,len(values)-7,8)]
        available=list(groups)
        while len(available)>=8:
            chosen=rng.sample(available,8)
            batch=[]
            for identity in chosen:
                batch.extend(queues[identity].pop(0))
                if not queues[identity]:available.remove(identity)
            batches.append(batch)
    return batches

def queue_read(cache,step,indices):
    for i in list(cache):
        if step-cache[i]>8:del cache[i]
    return [dict(record_index=i,identity=records[i]['identity'],scene=records[i]['scene'],age=step-stored,stored_step=stored) for i,stored in cache.items() if i not in set(indices)]

def queue_update(cache,step,indices):
    for i in indices:
        cache.pop(i,None);cache[i]=step
    while len(cache)>512:del cache[next(iter(cache))]

for fold,md,zero in zip(protocol['folds'],meta['folds'],t0['folds'],strict=True):
    f=fold['fold'];expected=sorted(set(eligible[f::3]+single[f::3]))
    assert fold['heldout_ids']==expected
    assert fold['source_ids']==sorted(set(identities)-set(expected))
    assert fold['source_label_map']=={str(i):j for j,i in enumerate(fold['source_ids'])}
    assert set(fold['source_record_indices']).isdisjoint(fold['gallery_record_indices'])
    assert set(fold['source_record_indices'])|set(fold['gallery_record_indices'])==record_ids
    assert fold['source_record_indices']==[r['index'] for r in records if r['identity'] in set(fold['source_ids'])]
    gallery=[records[i] for i in fold['gallery_record_indices']]
    queries=[];excluded=[]
    for position,r in enumerate(gallery):
        positive=sum(x['identity']==r['identity'] and x['scene']!=r['scene'] for x in gallery)
        removed=sum(x['identity']==r['identity'] and x['scene']==r['scene'] for x in gallery)
        if positive:
            queries.append(dict(record_index=r['index'],gallery_position=position,identity=r['identity'],scene=r['scene'],valid_positives=positive,removed_same_identity_same_scene=removed,retained_gallery=len(gallery)-removed,negative_identity_distractors=sum(x['identity']!=r['identity'] for x in gallery)))
        else:excluded.append(r['index'])
    assert queries==fold['query_rows'] and excluded==fold['excluded_query_record_indices']
    batches=sampled_batches(fold);sampler_batches[f]=batches
    assert len(batches)==260 and batches==[x['record_indices'] for x in md['batches']]
    cache={};counts=[];max_age=0
    for step,(indices,z) in enumerate(zip(batches,zero['steps'],strict=True)):
        assert len(indices)==64 and sorted(collections.Counter(records[i]['identity'] for i in indices).values())==[8]*8
        history=queue_read(cache,step,indices);counts.append(len(history))
        ages=[r['age'] for r in history]
        assert z==dict(step=step+1,historical_records=len(history),ages=ages)
        if ages:max_age=max(max_age,max(ages))
        if step>=65:queue_update(cache,step,indices)
    assert sum(counts)==zero['total_historical_candidate_records'] and max(counts)==zero['maximum_historical_records']
    assert sum(x>0 for x in counts)==zero['active_steps']
    fold_results.append(dict(fold=f,source_ids=len(fold['source_ids']),source_records=len(fold['source_record_indices']),heldout_ids=len(expected),gallery_records=len(gallery),query_records=len(queries),sampler_batches=260,t0_candidates=sum(counts),t0_max_history=max(counts),t0_max_age=max_age))

def close(a,b,tolerance=2e-6):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<tolerance,(a,b,tolerance)
    return abs(a-b)

def norm_arithmetic(g):
    a,b,d=g['first_norm'],g['second_norm'],g['difference_norm'];c=g['cosine']
    assert all(math.isfinite(v) and v>=0 for v in (a,b,d))
    assert (c is None)==(a==0 or b==0)
    if c is not None:
        assert abs(c)<=1.00001
        assert abs(d*d-(a*a+b*b-2*a*b*c))<=1e-7*max(1,a*a+b*b)

details=[];endpoint_results=[];pairs={};direct_rows=[];all_indices=set()
for folder in [f'fold_{f}_{e}' for f in range(3) for e in ('control','role_set')]+['overfit_control','overfit_role_set']:
    overfit=folder.startswith('overfit');fold_index=0 if overfit else int(folder.split('_')[1]);endpoint='role_set' if folder.endswith('role_set') else 'control'
    fold=protocol['folds'][fold_index];n=100 if overfit else 8
    directory=RUN/'m0'/folder
    training=read(directory/'training.json')
    audits=[json.loads(line) for line in (directory/'memory_steps.jsonl').read_text().splitlines()]
    target=summary['overfit'][endpoint] if overfit else summary['folds'][fold_index]['endpoints'][endpoint]
    assert training==target['training'] and len(audits)==len(training['steps'])==training['optimizer_steps']==n
    if not overfit:assert read(directory/'receipt.json')==target
    assert training['epochs']==1 and training['overflow_events']==0
    assert training['trainable_tensors']==training['nonzero_gradient_tensors']==203 and not training['missing_nonzero_gradients']
    assert training['initial_state_sha256']!=training['final_state_sha256']
    assert training['frozen_state_before_sha256']==training['frozen_state_after_sha256']
    assert training['signal_state_before_sha256']==training['signal_state_after_sha256']
    assert target['initialization']['initial_state_sha256']==training['initial_state_sha256']
    train_names=target['initialization']['trainable_names']
    assert len(train_names)==203 and len([x for x in train_names if x.startswith('encoder.')])==189
    assert not target['initialization']['role_weights_loaded']
    assert target['initialization']['source_ids']==fold['source_ids'] and target['initialization']['heldout_ids']==fold['heldout_ids']
    cache={};elements=0;fresh_count=64;vjp_count=0;direct_count=0;max_loss_error=0;max_total_error=0;max_history=0;max_age=0;hist_candidates=0;extras=0;active_extras=0;unique=set();pair=[]
    with (directory/'memory_distances.f32').open('rb') as stream:
        for step,(row,audit) in enumerate(zip(training['steps'],audits,strict=True)):
            assert row['step']==audit['step']==step+1 and audit['zero_based_step']==step and row['epoch']==1
            indices=sampler_batches[fold_index][0 if overfit else step]
            assert indices==row['sampled_record_indices']==audit['record_indices']
            assert set(indices)<=set(fold['source_record_indices'])
            ids=[records[i]['identity'] for i in indices];scenes=[records[i]['scene'] for i in indices]
            assert ids==audit['identities'] and scenes==audit['scenes']
            unique.update(indices);all_indices.update(indices)
            expected=queue_read(cache,step,indices)
            assert expected==audit['memory']
            assert audit['warmup_steps']==2 and audit['replacement_active']==(step>=2)
            assert audit['history_anchor_count']==0 and audit['current_anchor_count']==64
            assert audit['coordinate_rule']=='fresh'
            count=len(expected);width=64+count;size=4*64*width
            assert stream.tell()==audit['distance_offset_bytes'] and size==audit['distance_float_count']
            raw=np.fromfile(stream,dtype='<f4',count=size)
            assert raw.size==size and np.isfinite(raw).all() and (raw>=0).all() and (raw<=2.00001).all()
            matrices=raw.reshape(4,64,width);allids=ids+[x['identity'] for x in expected]
            proposal_rows=[];negative_counts=[];extra_active=[];basic=[];hard=[];candidate=[]
            stats=collections.Counter()
            for anchor in range(64):
                pos=[j for j in range(64) if j!=anchor and ids[j]==ids[anchor]]
                neg=[j for j in range(64) if ids[j]!=ids[anchor]]
                hp=max(float(matrices[0,anchor,j]) for j in pos)
                hn=min(float(matrices[0,anchor,j]) for j in neg)
                history_pos=[j for j in range(64,width) if allids[j]==ids[anchor]]
                history_neg=[j for j in range(64,width) if allids[j]!=ids[anchor]]
                full_hp=max([hp]+[float(matrices[0,anchor,j]) for j in history_pos])
                full_hn=min([hn]+[float(matrices[0,anchor,j]) for j in history_neg])
                negative=neg+history_neg
                proposed=[min(negative,key=lambda j:float(matrices[k,anchor,j])) for k in range(4)]
                selected=set(proposed);extra=selected-{proposed[0]}
                base_hinge=max(0.,hp-hn+.3);hard_hinge=max(0.,full_hp-full_hn+.3)
                extra_hinges=[max(0.,full_hp-float(matrices[0,anchor,j])+.3) for j in extra]
                candidate.append((hard_hinge+sum(extra_hinges))/len(selected))
                basic.append(base_hinge);hard.append(hard_hinge)
                proposal_rows.append(proposed);negative_counts.append(len(selected));extra_active.append(sum(x>0 for x in extra_hinges))
                stats.update(memory_positive_pairs=len(history_pos),memory_negative_pairs=len(history_neg),memory_cross_scene_positive_pairs=sum(expected[j-64]['scene']!=scenes[anchor] for j in history_pos),memory_negative_violations_against_batch_hard_positive=sum(float(matrices[0,anchor,j])<hp+.3 for j in history_neg),harder_positive_anchors=int(full_hp>hp),harder_negative_anchors=int(full_hn<hn),current_wrong_order_anchors=int(hp>=hn),expanded_wrong_order_anchors=int(full_hp>=full_hn),expanded_hinge_positive_anchors=int(hard_hinge>0))
            b=sum(basic)/64;h=sum(hard)/64;r=sum(candidate)/64
            selection=audit['relation_objective']
            assert proposal_rows==selection['proposals'] and negative_counts==selection['negative_counts'] and extra_active==selection['extra_active_counts']
            assert audit['proposal_order']==['fused',*EXPERTS]
            for key,value in stats.items():assert audit['statistics'][key]==value,(folder,step,key)
            ages=[x['age'] for x in expected];age=max(ages,default=0)
            assert audit['statistics']['memory_records']==count and audit['statistics']['maximum_memory_age']==age
            components=row['components'];wanted_keys={'id_fused','triplet_fused'}|{k+'_'+e for k in ('id','triplet','id_residual','triplet_residual') for e in EXPERTS}
            assert set(components)==wanted_keys and all(math.isfinite(x) for x in components.values())
            expected_loss=b if step<2 else h if endpoint=='control' else r
            errors=[close(x,y) for x,y in ((audit['original_triplet'],b),(audit['statistics']['current_triplet'],b),(audit['statistics']['expanded_triplet'],h),(selection['hard_loss'],h),(selection['role_set_loss'],r),(components['triplet_fused'],expected_loss))]
            w=config['LOSS']
            total=w['ID_FUSED']*components['id_fused']+w['TRIPLET_FUSED']*components['triplet_fused']
            total+=sum(w['ID_BRANCH']*components['id_'+e]+w['TRIPLET_BRANCH']*components['triplet_'+e]+w['ID_RESIDUAL']*components['id_residual_'+e]+w['TRIPLET_RESIDUAL']*components['triplet_residual_'+e] for e in EXPERTS)
            total_error=close(total,row['loss'],1e-5)
            assert row['amp_scale_before']>0 and row['amp_scale_after']>=row['amp_scale_before']
            group_count=64*len({x['stored_step'] for x in expected})
            assert group_count==audit['fresh_role_record_forwards'];fresh_count+=group_count
            norms=audit['historical_leaf_upstream_norms']
            assert len(norms)==count and all(math.isfinite(x) and x>=0 for x in norms)
            groups=sorted({x['stored_step'] for x,norm in zip(expected,norms,strict=True) if norm>0})
            assert groups==audit['history_vjp_groups'] and 64*len(groups)==audit['history_vjp_record_forwards']
            vjp_count+=64*len(groups)
            assert audit['gradient_weight']==1 and audit['history_candidate_vjp_applied']
            assert all(audit[k] for k in ('selected_reencoding_bitwise','history_rng_buffers_preserved','history_vjp_leaves_current_grad_unchanged','final_gradient_addition_bitwise'))
            for role in EXPERTS:
                partial=audit['roles'][role]['total_vs_history'];both=audit['roles'][role]['total_vs_both'];applied=audit['applied_gradients'][role]
                for g in (partial,both,applied):norm_arithmetic(g)
                close(both['difference_norm'],partial['second_norm'],1e-6*max(1,partial['second_norm']))
                for key in both:
                    if both[key] is None:assert applied[key] is None
                    else:close(both[key],applied[key],1e-6*max(1,abs(both[key])))
            direct=audit['direct_single_group_check']
            if direct:
                assert not overfit and len({x['stored_step'] for x in expected})==1
                assert direct['all_four_reencoded_outputs_bitwise_equal']
                norm_arithmetic(direct)
                ratio=direct['difference_norm']/max(direct['first_norm'],1e-12)
                close(ratio,direct['relative_l2_error'],1e-12);assert ratio<=.005
                direct_rows.append(dict(endpoint=folder,step=step+1,encoder_tensors=189,**direct));direct_count+=1
            if step>=2:queue_update(cache,step,indices)
            elements+=size;hist_candidates+=count;max_history=max(max_history,count);max_age=max(max_age,age);extras+=sum(n-1 for n in negative_counts);active_extras+=sum(extra_active);max_loss_error=max(max_loss_error,max(errors));max_total_error=max(max_total_error,total_error)
            pair.append((indices,audit['pixel_sha256']))
            details.append(dict(endpoint=folder,step=step+1,matrix_elements=size,history=count,unique_current=len(set(indices)),max_age=age,basic_triplet=b,hard_triplet=h,role_set_mean=r,extra_negative_positions=sum(n-1 for n in negative_counts),active_extra_hinges=sum(extra_active),total_loss_recomputed=total,total_loss_recorded=row['loss'],loss_max_error=max(errors),total_loss_error=total_error,history_vjp_groups=groups,direct_check_present=bool(direct)))
        assert stream.read()==b''
    assert fresh_count==training['extra_fresh_role_record_forwards'] and vjp_count==training['extra_history_vjp_record_forwards']
    assert direct_count==(0 if overfit else 1) and direct_count*64==training['extra_direct_check_record_forwards']
    history=training['history'];assert len(history)==1 and history[0]['optimizer_steps']==n
    close(history[0]['mean_loss'],sum(r['loss'] for r in training['steps'])/n,1e-12)
    close(history[0]['learning_rate'],.00035,1e-12)
    loss_gate=None
    if overfit:
        classes=len(fold['source_ids']);correct=1-.1+.1/classes;other=.1/classes
        floor=-(correct*math.log(correct)+(classes-1)*other*math.log(other))*.75
        first,last=training['steps'][0]['loss'],training['steps'][-1]['loss']
        ratio=(last-floor)/(first-floor)
        close(floor,target['gate']['minimum_loss'],1e-12);close(ratio,target['gate']['loss_ratio'],1e-12)
        assert ratio<=.1 and hist_candidates==0
        loss_gate=dict(classes=classes,weighted_CE_floor=floor,first_loss=first,last_loss=last,excess_ratio=ratio,history_candidates=0)
    key='overfit' if overfit else str(fold_index)
    pairs.setdefault(key,[]).append(pair)
    endpoint_results.append(dict(endpoint=folder,steps=n,unique_records=len(unique),trainable_tensors=203,encoder_tensors=189,matrix_elements=elements,history_candidates=hist_candidates,max_history=max_history,max_age=max_age,extra_negative_positions=extras,active_extra_hinges=active_extras,extra_fresh_role_record_forwards=fresh_count,extra_history_vjp_record_forwards=vjp_count,direct_checks=direct_count,peak_allocated_mib=training['peak_allocated_mib'],peak_reserved_mib=training['peak_reserved_mib'],training_elapsed_seconds=history[0]['elapsed_seconds'],max_loss_error=max_loss_error,max_total_loss_error=max_total_error,overfit=loss_gate))
assert all(a==b for a,b in pairs.values())
assert summary['optimizer_steps']==sum(x['steps'] for x in endpoint_results)==248
assert summary['heldout_record_forwards']==summary['official_image_reads']==0
assert len(direct_rows)==6 and sum(x['matrix_elements'] for x in endpoint_results)==4945920
assert sum(x['extra_history_vjp_record_forwards'] for x in endpoint_results)==5760
result={'status':'PASS_INDEPENDENT_M0_SAVED_ARRAY_ARITHMETIC','numpy_version':np.__version__,'cpu_thread_environment':{k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},'fold_provenance':fold_results,'dataset_filename_records':1032,'dataset_filename_paths':3096,'dataset_image_byte_reads':0,'sampler_batches_reconstructed':780,'m0_steps':248,'m0_unique_records':len(all_indices),'matrix_elements':sum(x['matrix_elements'] for x in endpoint_results),'paired_records_and_pixel_sha_equal':True,'endpoint_checks':endpoint_results,'direct_recorded_norm_checks':direct_rows,'steps':details,'elapsed_seconds':time.time()-START,'limits':['Saved distances rechecked; model features and derivatives not regenerated.','Pixel hashes compared between paired logs; augmented tensors were not saved and pixels were not regenerated.','Direct-gradient vectors and original RNG/cache tensors are not persisted; norm arithmetic is checkable, vector equality remains a run assertion.','Full M0 overfit has no historical candidates; history VJP coverage is from capacity endpoints.'],'model_forwards':0,'model_backwards':0,'q1_artifacts_read':0,'official_image_reads':0}
print(json.dumps(result,ensure_ascii=False,indent=2))
