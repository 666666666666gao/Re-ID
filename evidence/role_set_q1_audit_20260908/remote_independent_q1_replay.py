"""Fresh auditor implementation. No project evaluator/verifier/model imports.

Read-only CPU evidence replay, with two computation threads. The saved training
geometry can validate objective arithmetic but cannot recreate model gradients.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,hashlib,math,time,datetime,collections
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
R=pathlib.Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=pathlib.Path('/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739')
Q=RUN/'q1'
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
    digest=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):digest.update(b)
    return digest.hexdigest()
def statehash(state):
    digest=hashlib.sha256()
    for n in sorted(state):digest.update(n.encode());digest.update(state[n].contiguous().numpy().tobytes())
    return digest.hexdigest()
def near(a,b,tol=1e-10):assert abs(float(a)-float(b))<=tol,(a,b,tol)
def stats(v):
    return dict(n=len(v),min=float(min(v)),max=float(max(v)),mean=float(np.mean(v))) if len(v) else dict(n=0)
started=time.monotonic();registered=read(R/'configs/MSVR310/TriFusion-role-set-paired-v1.json')
config=read(R/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
base=read(R/config['BASELINE']['CONFIG']);protocol=read(R/base['protocol'])
metadata=read(R/config['SOURCE_METADATA']['PATH']);b0=read(config['BASELINE']['SUMMARY'])
summary=read(Q/'summary.json');cpu=read(RUN/'q1_cpu.json');pipeline=read(RUN/'pipeline.json')
assert sha(Q/'summary.json')==pipeline['terminal_summary_sha256']==cpu['summary_sha256']
assert sha(RUN/'q1_cpu.json')==pipeline['terminal_cpu_sha256']
assert sha(R/'configs/MSVR310/TriFusion-role-set-paired-v1.json')==summary['config_sha256']
all_file_receipts=[]
for path,ref in cpu['files'].items():
    p=pathlib.Path(path);actual=sha(p)
    assert p.stat().st_size==ref['bytes'] and actual==ref['sha256'],path
    all_file_receipts.append(dict(path=path,bytes=p.stat().st_size,sha256=actual))
rows=protocol['records'];assert len(rows)==1032
names=[];dataset=pathlib.Path(base['dataset_root'])
for index,r in enumerate(rows):
    assert r['index']==index
    triplet=[pathlib.Path(p) for p in r['paths']]
    assert [p.parts[0] for p in triplet]==['bounding_box_train']*3
    assert [p.parts[2] for p in triplet]==['vis','ni','th']
    assert len({p.name for p in triplet})==1
    name=triplet[0].name;names.append(name)
    assert (int(name[:4]),int(name[6:9]),int(name[11]))==(r['identity'],r['scene'],r['camera'])
    assert all(int(p.parts[1])==r['identity'] and (dataset/p).is_file() for p in triplet)
assert len(set(names))==len(names)
listed=sorted(p.relative_to(dataset).as_posix() for p in (dataset/'bounding_box_train').glob('*/vis/*') if p.is_file())
assert sorted(r['paths'][0] for r in rows)==listed
ids=sorted({r['identity'] for r in rows});scene_map={i:{r['scene'] for r in rows if r['identity']==i} for i in ids}
eligible=[i for i in ids if len(scene_map[i])>1];ineligible=[i for i in ids if len(scene_map[i])==1]
assert (len(ids),len(eligible),len(ineligible))==(155,60,95)
reconstructed=[set() for _ in range(3)]
for group in (eligible,ineligible):
    for i,identity in enumerate(group):reconstructed[i%3].add(identity)
outputs=('baseline_only','fused','cnn','transformer','mamba');experts=outputs[2:]
widths=dict(zip(outputs,[3072,7680,4608,4608,4608]))
components=['id_fused','triplet_fused']+[k+'_'+e for e in experts for k in ('id','triplet','id_residual','triplet_residual')]
weight={k:config['LOSS']['ID_FUSED' if k=='id_fused' else 'TRIPLET_FUSED' if k=='triplet_fused' else 'ID_RESIDUAL' if k.startswith('id_residual') else 'TRIPLET_RESIDUAL' if k.startswith('triplet_residual') else 'ID_BRANCH' if k.startswith('id_') else 'TRIPLET_BRANCH'] for k in components}
endpoint_reports=[];fold_reports=[];all_ap={e:{o:[] for o in outputs} for e in ('control','role_set')};all_first={e:{o:[] for o in outputs} for e in ('control','role_set')}
query_ids=[];distance_total=0;training_elements=0;paired_rows=[]
for fold,fr,md,b in zip(protocol['folds'],summary['folds'],metadata['folds'],b0['folds'],strict=True):
    fi=fold['fold'];assert fi==fr['fold']==md['fold']==b['fold']
    assert set(fold['heldout_ids'])==reconstructed[fi]
    assert fold['source_ids']==sorted(set(ids)-reconstructed[fi])
    src=[r['index'] for r in rows if r['identity'] in set(fold['source_ids'])]
    gal=[r['index'] for r in rows if r['identity'] in reconstructed[fi]]
    assert src==fold['source_record_indices'] and gal==fold['gallery_record_indices']
    assert not set(src)&set(gal) and sorted(src+gal)==list(range(1032))
    assert fold['source_label_map']=={str(i):n for n,i in enumerate(fold['source_ids'])}
    gallery=[rows[i] for i in gal];gid=np.array([r['identity'] for r in gallery]);gsc=np.array([r['scene'] for r in gallery])
    positions=[];queries=[];excluded=[]
    for gp,r in enumerate(gallery):
        positives=int(((gid==r['identity'])&(gsc!=r['scene'])).sum());removed=int(((gid==r['identity'])&(gsc==r['scene'])).sum())
        if positives:
            positions.append(gp);queries.append(dict(record_index=r['index'],gallery_position=gp,identity=r['identity'],scene=r['scene'],valid_positives=positives,removed_same_identity_same_scene=removed,retained_gallery=len(gal)-removed,negative_identity_distractors=int((gid!=r['identity']).sum())))
        else:excluded.append(r['index'])
    assert queries==fold['query_rows'] and excluded==fold['excluded_query_record_indices']
    query_ids.extend(gid[positions].tolist())
    bp=torch.load(b['checkpoint'],map_location='cpu',weights_only=True)
    assert bp['source_ids']==fold['source_ids'] and bp['heldout_ids']==fold['heldout_ids'] and bp['fold']==fi
    bs=bp['model_state_dict'];baseline_arrays=torch.load(pathlib.Path(b['checkpoint']).parent/'retrieval_arrays.pt',map_location='cpu',weights_only=True)
    fold_scores={};train_pair=[];audit_pair=[]
    for endpoint in ('control','role_set'):
        d=Q/f'fold_{fi}_{endpoint}';item=fr['endpoints'][endpoint];tr=read(d/'training.json')
        assert item==read(d/'receipt.json') and item['training']==tr
        audits=[json.loads(line) for line in (d/'memory_steps.jsonl').read_text().splitlines()]
        assert tr['epochs']==20 and tr['optimizer_steps']==len(tr['steps'])==len(audits)==len(md['batches'])==260
        assert tr['overflow_events']==0 and tr['missing_nonzero_gradients']==[]
        assert tr['trainable_tensors']==tr['nonzero_gradient_tensors']==203
        assert tr['initial_state_sha256']==item['initialization']['initial_state_sha256']
        assert tr['initial_state_sha256']!=tr['final_state_sha256']
        assert tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256']
        assert tr['signal_state_before_sha256']==tr['signal_state_after_sha256']==statehash(bs)
        assert len(item['initialization']['trainable_names'])==203
        encoder_count=sum(n.startswith('encoder.') for n in item['initialization']['trainable_names'])
        assert encoder_count==189
        cp=torch.load(d/'roles_epoch20.pth',map_location='cpu',weights_only=True)
        assert cp['config_sha256']==summary['config_sha256'] and cp['binding']==item['initialization']
        assert cp['source_ids']==fold['source_ids'] and cp['heldout_ids']==fold['heldout_ids']
        state={n:bs[alias] for n,alias in cp['baseline_aliases'].items()};state.update(cp['role_state_dict'])
        assert statehash(state)==tr['final_state_sha256']==item['strict_reload_state_sha256']
        train_names=set(item['initialization']['trainable_names'])
        # The seven BatchNorm necks have mutable running buffers. They are not
        # frozen parameters (signal_preserving_v8.py:550-611); frozen_state_sha
        # includes all baseline state and frozen neck biases, not these buffers.
        buffer_names={n for n in state if not n.startswith('baseline.') and n.rsplit('.',1)[-1] in ('running_mean','running_var','num_batches_tracked')}
        assert len(buffer_names)==21
        frozen={n:t for n,t in state.items() if n.startswith('baseline.') or n not in train_names and n not in buffer_names}
        # Hash format is key+raw tensor bytes; no model forward/reconstruction.
        assert statehash(frozen)==tr['frozen_state_after_sha256']
        assert not any('optimizer' in n or 'memory' in n or 'cache' in n for n in cp)
        cache={};serial=0;mem_lengths=[];ages=[];duplicates=[];queue_sizes=[];sets=[];active_extras=[];unique_negative_ids=[]
        hard_weight=[];hard_losses=[];set_losses=[];fresh=64;vjp=0;max_loss_error=0.;max_ledger_error=0.;first_history=None
        within_current_dups=0;different_scene_negative_proposals=0;proposals_history=0;role_only_positions=0
        stage_stats={k:{'hard':[],'set':[],'counts':[],'active_extra':[]} for k in ('warmup','post_warmup_current_only','history')}
        gradients={'runtime_assertion_rows':0,'nonzero_history_role_rows':{e:0 for e in experts},'direct_graph_checks':0}
        with (d/'memory_distances.f32').open('rb') as f:
            for i,(row,a,mb) in enumerate(zip(tr['steps'],audits,md['batches'],strict=True)):
                current=row['sampled_record_indices'];current_ids=[rows[k]['identity'] for k in current];current_scenes=[rows[k]['scene'] for k in current]
                assert current==a['record_indices']==mb['record_indices'] and set(current)<=set(src)
                assert row['step']==a['step']==i+1 and a['zero_based_step']==i
                assert row['epoch']==i//13+1
                assert a['identities']==current_ids and a['scenes']==current_scenes
                assert sorted(collections.Counter(current_ids).values())==[8]*8
                duplicates.append(64-len(set(current)));within_current_dups+=64-len(set(current))
                cache={k:v for k,v in cache.items() if i-v[0]<=8}
                expected=[]
                for index,(step,order) in sorted(cache.items(),key=lambda kv:kv[1][1]):
                    if index not in set(current):expected.append(dict(record_index=index,identity=rows[index]['identity'],scene=rows[index]['scene'],age=i-step,stored_step=step))
                assert expected==a['memory'];m=len(expected);mem_lengths.append(m);ages.extend(r['age'] for r in expected)
                if m and first_history is None:first_history=i+1
                assert a['replacement_active']==(i>=65) and a['warmup_steps']==65
                assert a['history_anchor_count']==0 and a['current_anchor_count']==64 and a['coordinate_rule']=='fresh'
                assert a['proposal_order']==['fused',*experts]
                assert f.tell()==a['distance_offset_bytes'];n=4*64*(64+m)
                raw=np.fromfile(f,dtype='<f4',count=n);assert len(raw)==n==a['distance_float_count']
                assert np.isfinite(raw).all() and raw.min()>=0
                training_elements+=n;D=raw.reshape(4,64,64+m)
                ident=np.array(current_ids);scenes=np.array(current_scenes);mids=np.array([r['identity'] for r in expected],dtype=int);ms=np.array([r['scene'] for r in expected],dtype=int)
                cid=np.concatenate((ident,mids));csc=np.concatenate((scenes,ms))
                same=ident[:,None]==cid[None,:];pos=same.copy();pos[np.arange(64),np.arange(64)]=False;neg=~same
                curpos=pos[:,:64];curneg=neg[:,:64]
                batch_hp=np.max(np.where(curpos,D[0,:,:64],-np.inf),axis=1);batch_hn=np.min(np.where(curneg,D[0,:,:64],np.inf),axis=1)
                hp=np.max(np.where(pos,D[0],-np.inf),axis=1);hn=np.min(np.where(neg,D[0],np.inf),axis=1)
                # Independent per-anchor union over actual proposal positions.
                prop=np.argmin(np.where(neg[None,:,:],D,np.inf),axis=2).T
                anchor_hard=np.maximum(hp-hn+np.float32(.3),0);hinges=np.maximum(hp[:,None]-D[0]+np.float32(.3),0)
                cnt=[];extra=[];per_set=[]
                for ai in range(64):
                    chosen=set(map(int,prop[ai]));additional=chosen-{int(prop[ai,0])}
                    assert all(cid[g]!=ident[ai] for g in chosen)
                    cnt.append(len(chosen));extra.append(sum(hinges[ai,g]>0 for g in additional))
                    per_set.append((float(anchor_hard[ai])+sum(float(hinges[ai,g]) for g in additional))/len(chosen))
                    unique_negative_ids.append(len({int(cid[g]) for g in chosen}))
                    different_scene_negative_proposals+=sum(csc[g]!=scenes[ai] for g in chosen)
                    proposals_history+=sum(g>=64 for g in chosen);role_only_positions+=len(additional)
                basic=float(np.maximum(batch_hp-batch_hn+np.float32(.3),0).mean());hard=float(anchor_hard.mean());role=float(np.mean(per_set))
                rr=a['relation_objective'];assert prop.tolist()==rr['proposals'] and cnt==rr['negative_counts'] and extra==rr['extra_active_counts']
                for actual,calculated in [(a['original_triplet'],basic),(rr['hard_loss'],hard),(rr['role_set_loss'],role),(row['components']['triplet_fused'],(hard if endpoint=='control' else role) if i>=65 else basic)]:
                    error=abs(actual-calculated);max_loss_error=max(max_loss_error,error);assert error<2e-6
                assert role<=hard+2e-6
                expected_stats={'memory_records':m,'memory_positive_pairs':int(same[:,64:].sum()),'memory_negative_pairs':int(neg[:,64:].sum()),'memory_cross_scene_positive_pairs':int((same[:,64:]&(scenes[:,None]!=ms)).sum()),'memory_negative_violations_against_batch_hard_positive':int((neg[:,64:]&(D[0,:,64:]<batch_hp[:,None]+np.float32(.3))).sum()),'harder_positive_anchors':int((hp>batch_hp).sum()),'harder_negative_anchors':int((hn<batch_hn).sum()),'current_wrong_order_anchors':int((batch_hp>=batch_hn).sum()),'expanded_wrong_order_anchors':int((hp>=hn).sum()),'expanded_hinge_positive_anchors':int((hp-hn+np.float32(.3)>0).sum()),'maximum_memory_age':max([r['age'] for r in expected],default=0)}
                assert all(a['statistics'][k]==v for k,v in expected_stats.items())
                near(a['statistics']['current_triplet'],basic,2e-6);near(a['statistics']['expanded_triplet'],hard,2e-6)
                assert set(row['components'])==set(components)
                ledger=sum(row['components'][k]*weight[k] for k in components);max_ledger_error=max(max_ledger_error,abs(ledger-row['loss']));near(ledger,row['loss'],1e-5)
                assert row['amp_scale_before']==row['amp_scale_after']==256
                norms=a['historical_leaf_upstream_norms'];assert len(norms)==m and all(math.isfinite(v) and v>=0 for v in norms)
                groups=sorted({r['stored_step'] for r,norm in zip(expected,norms) if norm>0});assert groups==a['history_vjp_groups']
                vjp_count=64*len(groups);fresh_count=64*len({r['stored_step'] for r in expected})
                assert vjp_count==a['history_vjp_record_forwards'] and fresh_count==a['fresh_role_record_forwards'];vjp+=vjp_count;fresh+=fresh_count
                for flag in ['history_candidate_vjp_applied','selected_reencoding_bitwise','history_rng_buffers_preserved','history_vjp_leaves_current_grad_unchanged','final_gradient_addition_bitwise']:assert a[flag]
                gradients['runtime_assertion_rows']+=1
                assert not a['direct_single_group_check'];near(a['gradient_weight'],1.)
                for e in experts:
                    g=a['roles'][e];applied=a['applied_gradients'][e]
                    for obj in [g['total_vs_history'],g['total_vs_both'],applied]:
                        u,v,diff=obj['first_norm'],obj['second_norm'],obj['difference_norm'];c=obj['cosine']
                        assert all(math.isfinite(z) and z>=0 for z in [u,v,diff]);assert (c is None)==(u==0 or v==0)
                        if c is not None:near(diff*diff,u*u+v*v-2*u*v*c,1e-7*max(1,u*u+v*v));assert abs(c)<=1.00001
                    near(g['total_vs_both']['difference_norm'],g['total_vs_history']['second_norm'],1e-6*max(1,g['total_vs_history']['second_norm']))
                    for k,v in applied.items():
                        expected_v=g['total_vs_both'][k]
                        if v is None:assert expected_v is None
                        else:near(v,expected_v,1e-6*max(1,abs(v),abs(expected_v)))
                    gradients['nonzero_history_role_rows'][e]+=int(g['total_vs_history']['second_norm']>0)
                phase='warmup' if i<65 else 'post_warmup_current_only' if m==0 else 'history'
                ss=stage_stats[phase];ss['hard'].append(hard);ss['set'].append(role);ss['counts'].extend(cnt);ss['active_extra'].extend(extra)
                sets.extend(cnt);active_extras.extend(extra);hard_weight.extend(1/n for n in cnt);hard_losses.append(hard);set_losses.append(role)
                if i>=65:
                    for index in current:serial+=1;cache[index]=(i,serial)
                    while len(cache)>512:del cache[min(cache,key=lambda k:cache[k][1])]
                queue_sizes.append(len(cache))
            assert not f.read(1)
        assert fresh==tr['extra_fresh_role_record_forwards'] and vjp==tr['extra_history_vjp_record_forwards'] and tr['extra_direct_check_record_forwards']==0
        for epoch,er in enumerate(tr['history'],1):
            assert er['epoch']==epoch and er['optimizer_steps']==13
            mult=epoch/5 if epoch<=5 else .5*(1+math.cos(math.pi*(epoch-6)/15))
            near(er['learning_rate'],.00035*mult,1e-15);near(er['mean_loss'],np.mean([v['loss'] for v in tr['steps'][(epoch-1)*13:epoch*13]]),1e-14)
        ar=torch.load(d/'retrieval_arrays.pt',map_location='cpu',weights_only=True);ranks=read(d/'rankings.json')
        assert ar['gallery_record_indices']==gal and ar['query_gallery_positions']==positions
        assert item['retrieval']['gallery_manifest']==gallery and item['retrieval']['query_rows']==queries
        assert torch.equal(ar['features']['baseline_only'],baseline_arrays['features']) and torch.equal(ar['distances']['baseline_only'],baseline_arrays['distances'])
        assert torch.equal(ar['features']['fused'][:,:3072],ar['features']['baseline_only'])
        for e in experts:assert torch.equal(ar['features'][e][:,:3072],ar['features']['baseline_only'])
        scores={};retrieval_checks=[]
        for o in outputs:
            feature=ar['features'][o];distance=ar['distances'][o].numpy()
            assert feature.shape==(len(gal),widths[o]) and torch.isfinite(feature).all()
            assert distance.shape==(len(positions),len(gal)) and np.isfinite(distance).all()
            # Float64 formula is independent of recorded torch.addmm FP32 execution.
            f64=feature.numpy().astype(np.float64);unit=f64/np.linalg.norm(f64,axis=1,keepdims=True)
            d64=np.sum(unit[positions]**2,axis=1)[:,None]+np.sum(unit**2,axis=1)[None,:]-2*unit[positions]@unit.T
            derr=float(np.max(np.abs(d64-distance)));assert derr<3e-6,(fi,endpoint,o,derr)
            order=np.argsort(distance,axis=1);assert order.tolist()==ranks[o]
            AP=[];first=[];tie_pairs=0;ties_with_mixed_labels=0
            for qi,sort in enumerate(order):
                qp=positions[qi];legal=[int(g) for g in sort if not(gid[g]==gid[qp] and gsc[g]==gsc[qp])]
                match_ranks=[i+1 for i,g in enumerate(legal) if gid[g]==gid[qp]]
                assert match_ranks
                AP.append(sum((i+1)/v for i,v in enumerate(match_ranks))/len(match_ranks));first.append(match_ranks[0])
                for g1,g2 in zip(legal,legal[1:]):
                    if distance[qi,g1]==distance[qi,g2]:tie_pairs+=1;ties_with_mixed_labels+=int((gid[g1]==gid[qp])!=(gid[g2]==gid[qp]))
            score=item['retrieval']['outputs'][o];assert np.max(np.abs(np.array(AP)-score['average_precision']))<1e-14 and first==score['first_match_rank']
            metrics={'mAP':float(np.mean(AP)*100),**{f'Rank-{k}':float(np.mean(np.array(first)<=k)*100) for k in [1,5,10]}}
            for k,v in metrics.items():near(v,score['metrics'][k])
            scores[o]=metrics;all_ap[endpoint][o].extend(AP);all_first[endpoint][o].extend(first);distance_total+=distance.size
            retrieval_checks.append(dict(output=o,shape=list(feature.shape),rank_elements=distance.size,float64_distance_max_abs_error=derr,exact_saved_rank_order=True,query_ap_and_first_rank_exact_to_1e14=True,legal_distance_tie_pairs=tie_pairs,mixed_label_distance_tie_pairs=ties_with_mixed_labels))
        fold_scores[endpoint]=scores
        phases={phase:{k:stats(v) for k,v in data.items()} for phase,data in stage_stats.items()}
        means={phase:{k:float(np.mean([s['loss'] if k=='total_loss' else s['components'][k] for i,s in enumerate(tr['steps']) if (i<65 if phase=='warmup' else i>=65 if phase=='post_warmup' else True)])) for k in ['total_loss',*components]} for phase in ['all','warmup','post_warmup']}
        endpoint_reports.append(dict(fold=fi,endpoint=endpoint,steps=260,source_record_exposures=260*64,current_record_duplicates=stats(duplicates),all_current_duplicate_exposures=within_current_dups,source_unique_records=len({r for step in tr['steps'] for r in step['sampled_record_indices']}),historical_records=stats(mem_lengths),historical_candidate_exposures=sum(mem_lengths),historical_age=stats(ages),queue_occupancy=stats(queue_sizes),first_history_step=first_history,objective_replacement_first_step=66,negative_set_count=stats(sets),distinct_negative_identity_count=stats(unique_negative_ids),mean_hard_term_relative_weight=float(np.mean(hard_weight)),extra_active_per_anchor=stats(active_extras),role_only_selected_positions=role_only_positions,selected_history_positions=proposals_history,selected_different_scene_negative_positions=different_scene_negative_proposals,hard_loss_mean=float(np.mean(hard_losses)),counterfactual_role_set_loss_mean=float(np.mean(set_losses)),objective_max_error=max_loss_error,total_ledger_max_error=max_ledger_error,phase_statistics=phases,loss_component_means=means,extra_fresh_role_record_forwards=fresh,extra_history_vjp_record_forwards=vjp,encoder_gradient_tensors=encoder_count,trainable_gradient_tensors=203,runtime_gradient_evidence=gradients,all_saved_training_rows_replayed=True,checkpoint_state_hash_reconstructed=True,frozen_endpoint_hash_reconstructed=True,initial_role_tensor_equality_independently_reconstructed=False,peak_reserved_mib=tr['peak_reserved_mib'],elapsed_epoch_seconds=sum(r['elapsed_seconds'] for r in tr['history']),retrieval_checks=retrieval_checks))
        train_pair.append(tr);audit_pair.append(audits)
        del cp,state,ar,feature
    # Keep pairing separate from independent evidence of original augmentation pixels.
    assert fr['endpoints']['control']['initialization']==fr['endpoints']['role_set']['initialization']
    assert [(r['record_indices'],r['pixel_sha256']) for r in audit_pair[0]]==[(r['record_indices'],r['pixel_sha256']) for r in audit_pair[1]]
    differences=[]
    for a,bx in zip(train_pair[0]['steps'],train_pair[1]['steps']):
        differences.append({'step':a['step'],'total_loss':bx['loss']-a['loss'],**{k:bx['components'][k]-a['components'][k] for k in components}})
    first_component_difference={k:next((r['step'] for r in differences if r[k]!=0),None) for k in ['total_loss',*components]}
    paired_rows.append(dict(fold=fi,all_sampled_records_and_pixel_hashes_match=True,initial_bindings_match=True,first_component_difference_step=first_component_difference,first65_max_abs_total_loss_difference=max(abs(r['total_loss']) for r in differences[:65]),first65_max_abs_component_difference=max(abs(r[k]) for r in differences[:65] for k in components),step66_total_loss_difference=differences[65]['total_loss']))
    fold_reports.append(dict(fold=fi,counts=fold['counts'],source_scene_values=fold['source_scene_values'],heldout_scene_values=fold['heldout_scene_values'],scores=fold_scores))
    del bp,bs,baseline_arrays
identities=np.array(query_ids);assert len(identities)==600 and len(set(identities))==60
aggregate={}
def boot(delta):
    clusters=sorted(set(query_ids));by={c:[i for i,x in enumerate(query_ids) if x==c] for c in clusters};rng=np.random.default_rng(42);results=[]
    for _ in range(10000):
        sampled=rng.choice(clusters,size=60,replace=True);positions=[i for c in sampled for i in by[c]]
        results.append(float(np.mean(delta[positions])))
    return float(np.percentile(results,2.5,method='linear'))
for e in ['control','role_set']:
    AP={o:np.array(all_ap[e][o]) for o in outputs};first={o:np.array(all_first[e][o]) for o in outputs};recorded=summary['comparison']['endpoints'][e]
    metrics={o:{'mAP':float(v.mean()*100),**{f'Rank-{k}':float(np.mean(first[o]<=k)*100) for k in [1,5,10]}} for o,v in AP.items()}
    gains={o:metrics[o]['mAP']-metrics['baseline_only']['mAP'] for o in outputs};fg=[f['scores'][e]['fused']['mAP']-f['scores'][e]['baseline_only']['mAP'] for f in fold_reports]
    lb=boot((AP['fused']-AP['baseline_only'])*100)
    checks=dict(fused_gain_at_least_1pp=gains['fused']>=1,all_fold_fused_gains_nonnegative=all(g>=0 for g in fg),all_full_branches_not_below_signal=all(gains[x]>=0 for x in experts),identity_bootstrap_lower_positive=lb>0,fused_strictly_best=all(metrics['fused']['mAP']>metrics[x]['mAP'] for x in ('baseline_only',*experts)))
    assert checks==recorded['scientific_checks'] and all(checks.values())==recorded['scientific_passed'];near(lb,recorded['identity_bootstrap']['lower_bound_pp'])
    for o in outputs:
        for k,v in metrics[o].items():near(v,recorded['metrics'][o][k])
        near(gains[o],recorded['gains_over_signal_pp'][o])
        for idrow in recorded['per_identity']:
            mask=identities==idrow['identity'];assert int(mask.sum())==idrow['query_count'];near(AP[o][mask].mean()*100,idrow['map_by_output'][o])
    aggregate[e]=dict(metrics=metrics,gains_over_signal_pp=gains,fold_fused_gains_pp=fg,identity_bootstrap_lower_pp=lb,checks=checks)
delta={o:(np.array(all_ap['role_set'][o])-np.array(all_ap['control'][o]))*100 for o in outputs};gains={o:float(v.mean()) for o,v in delta.items()}
fg=[f['scores']['role_set']['fused']['mAP']-f['scores']['control']['fused']['mAP'] for f in fold_reports];lb=boot(delta['fused']);m=aggregate['role_set']['metrics']
checks=dict(fused_gain_at_least_1pp=gains['fused']>=1,all_fold_fused_nonnegative=all(x>=0 for x in fg),all_role_gains_nonnegative=all(gains[x]>=0 for x in experts),paired_identity_bootstrap_lower_positive=lb>0,candidate_fused_strictly_best=all(m['fused']['mAP']>m[x]['mAP'] for x in ('baseline_only',*experts)))
comp=summary['comparison'];assert checks==comp['paired_checks'];near(lb,comp['paired_bootstrap_lower_pp'])
for o,v in gains.items():near(v,comp['matched_gains_mAP'][o])
for a,b in zip(fg,comp['fold_fused_gains_mAP']):near(a,b)
for r in comp['paired_per_identity']:
    mask=identities==r['identity'];assert r['query_count']==int(mask.sum())
    for o in outputs:near(delta[o][mask].mean(),r['gains_mAP'][o])
qualified=all(checks.values()) and all(aggregate['role_set']['checks'].values());assert qualified==comp['next_phase_qualified']
assert summary['status']==('Q1_PASS' if qualified else 'Q1_FAIL')
assert summary['optimizer_steps']==1560 and summary['heldout_record_forwards']==2064 and summary['official_image_reads']==0
assert not torch.cuda.is_initialized()
report=dict(status='PASS_INDEPENDENT_COMPLETE_Q1_SAVED_EVIDENCE_REPLAY',review_independence='deterministic',scope='saved evidence arithmetic and state hashes; no models/gradients/augmentation replay',generated_at=datetime.datetime.now().astimezone().isoformat(),numpy_version=np.__version__,torch_version=str(torch.__version__),cpu_threads=torch.get_num_threads(),model_forwards=0,optimizer_updates=0,image_reads=0,official_evaluations=0,training_steps=1560,training_distance_elements=training_elements,retrieval_distance_and_rank_elements=distance_total,source_protocol_reconstructed=True,all1032_dataset_filename_triplets_match=True,source_ids_total=155,eligible_query_ids=60,gallery_only_identities=95,folds=fold_reports,endpoints=endpoint_reports,pairing=paired_rows,aggregate=aggregate,paired_scientific=dict(gains_mAP=gains,fold_gains_mAP=fg,identity_bootstrap_lower_pp=lb,checks=checks),next_phase_qualified=qualified,scientific_status=summary['status'],files=all_file_receipts,elapsed_seconds=time.monotonic()-started)
print(json.dumps(report,ensure_ascii=False,allow_nan=False,default=lambda value:value.item()))
