"""Independent reviewer replay. Reads saved artifacts; no project imports/model execution/writes."""
import collections,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch

START=time.monotonic()
ROOT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/autodl-tmp/trifusion-v2/artifacts/msvr310_history_gradient_v1_seed42_a1b4777')
ENDS=('control','history_gradient'); OUTPUTS=('baseline_only','fused','cnn','transformer','mamba'); ROLES=OUTPUTS[2:]
HASHES={}
def read(path,kind='json'):
 p=Path(path); b=p.read_bytes(); HASHES[str(p)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 return json.loads(b) if kind=='json' else b
def digest(path):
 p=Path(path); h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 HASHES[str(p)]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
 return h.hexdigest()
def tensor_file(path):
 digest(path); return torch.load(path,map_location='cpu',weights_only=True)
def close(a,b,tol=1e-10):
 assert abs(float(a)-float(b))<=tol,(a,b,tol)
def mean(v):return math.fsum(map(float,v))/len(v)
def state_hash(state):
 h=hashlib.sha256()
 for k in sorted(state):h.update(k.encode());h.update(state[k].contiguous().numpy().tobytes())
 return h.hexdigest()

spec=read(ROOT/'configs/MSVR310/TriFusion-history-gradient-paired-v1.json')
config=read(ROOT/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
base=read(ROOT/'configs/MSVR310/Signal-source-oof-v1.json')
protocol=read(ROOT/base['protocol']); labels=read(ROOT/'evidence/vehicle_query_protocol_labels_20260905.json')
meta=read(ROOT/config['SOURCE_METADATA']['PATH'])
baseline=read(config['BASELINE']['SUMMARY'])
pipeline=read(RUN/'pipeline.json'); summary=read(RUN/'q1/summary.json'); cpu=read(RUN/'q1_cpu.json')
m0=read(RUN/'m0/summary.json'); m0cpu=read(RUN/'m0_cpu.json');t0=read(RUN/'t0.json')
assert [s['stage'] for s in pipeline['stages']]==['t0','m0','m0_cpu','q1','q1_cpu']
assert all(s['exit_code']==0 for s in pipeline['stages'])
assert pipeline['terminal_summary_sha256']==cpu['summary_sha256']==digest(RUN/'q1/summary.json')
assert pipeline['terminal_cpu_sha256']==digest(RUN/'q1_cpu.json')
assert summary['m0_receipt_sha256']==m0cpu['summary_sha256']==digest(RUN/'m0/summary.json')
assert summary['m0_verification_sha256']==digest(RUN/'m0_cpu.json')
assert pipeline['config_sha256']==summary['config_sha256']==m0['config_sha256']==digest(ROOT/'configs/MSVR310/TriFusion-history-gradient-paired-v1.json')
assert summary['official_image_reads']==m0['official_image_reads']==0

# Independently reconstruct the label-only deterministic partition from file names.
dataset=next(d for d in labels['datasets'] if d['dataset']=='MSVR310')
source=dataset['record_manifest']['bounding_box_train']
assert source==sorted(source,key=lambda r:r['path'])
records=[]
for i,r in enumerate(source):
 name=Path(r['path']).name; identity=int(name[:4]);scene=int(name[6:9]);camera=int(name[11])
 assert (identity,scene,camera)==(r['identity'],r['scene'],r['camera'])
 prefix=Path('bounding_box_train')/Path(r['path']).parts[0]
 records.append(dict(index=i,identity=identity,camera=camera,scene=scene,paths=[str(prefix/m/name) for m in ('vis','ni','th')]))
assert records==protocol['records']
identities=sorted({r['identity'] for r in records})
eligible=[i for i in identities if len({r['scene'] for r in records if r['identity']==i})>1]
single=[i for i in identities if i not in eligible]
part=[set(eligible[f::3]+single[f::3]) for f in range(3)]
fold_counts=[]
for f,fold in enumerate(protocol['folds']):
 assert fold['heldout_ids']==sorted(part[f])
 assert fold['source_ids']==sorted(set(identities)-part[f])
 assert fold['source_label_map']=={str(i):k for k,i in enumerate(fold['source_ids'])}
 src=[r['index'] for r in records if r['identity'] not in part[f]]
 gal=[r['index'] for r in records if r['identity'] in part[f]]
 assert src==fold['source_record_indices'] and gal==fold['gallery_record_indices']
 srcpaths={p for i in src for p in records[i]['paths']}; galpaths={p for i in gal for p in records[i]['paths']}
 assert not srcpaths & galpaths
 expected=[];excluded=[]
 for pos,i in enumerate(gal):
  q=records[i];removed=sum(records[j]['identity']==q['identity'] and records[j]['scene']==q['scene'] for j in gal)
  positive=sum(records[j]['identity']==q['identity'] and records[j]['scene']!=q['scene'] for j in gal)
  if not positive:excluded.append(i);continue
  expected.append(dict(record_index=i,gallery_position=pos,identity=q['identity'],scene=q['scene'],valid_positives=positive,removed_same_identity_same_scene=removed,retained_gallery=len(gal)-removed,negative_identity_distractors=sum(records[j]['identity']!=q['identity'] for j in gal)))
 assert expected==fold['query_rows'] and excluded==fold['excluded_query_record_indices']
 fold_counts.append(dict(fold=f,source_records=len(src),source_ids=len(fold['source_ids']),gallery_records=len(gal),queries=len(expected),query_ids=len({r['identity'] for r in expected}),excluded_query_records_retained_in_gallery=len(excluded)))
assert len(records)==1032 and len(identities)==155 and len(eligible)==60 and len(single)==95
assert sum(f['queries'] for f in fold_counts)==600
actual_paths=sorted(str(p.relative_to(Path(base['dataset_root']))) for p in (Path(base['dataset_root'])/'bounding_box_train').glob('*/*/*') if p.is_file())
assert actual_paths==sorted(p for r in records for p in r['paths'])
assert {r['identity'] for r in source}.isdisjoint({r['identity'] for r in dataset['record_manifest']['bounding_box_test']})
protocol_result=dict(records=1032,modal_paths=3096,identities=155,eligible_identities=60,distractor_identities=95,folds=fold_counts,all_full_paths_source_heldout_disjoint=True,real_training_filename_inventory_matches=True,official_image_contents_read=0)

training_results=[]; PAIRS={}; q1_rows={}; max_scalar=0.; max_triplet=0.; max_upstream=0.; total_distance=0
def training(mode,fold,end,item,dirname):
 global max_scalar,max_triplet,max_upstream,total_distance
 d=RUN/mode/dirname;tr=read(d/'training.json');assert tr==item['training']
 rows=[json.loads(x) for x in read(d/'memory_steps.jsonl','bytes').splitlines()]
 count=260 if mode=='q1' else 100 if dirname.startswith('overfit') else 8
 assert len(rows)==len(tr['steps'])==tr['optimizer_steps']==count
 for n,v in tr['audit_files'].items():assert digest(d/n)==v['sha256'] and (d/n).stat().st_size==v['bytes']
 assert tr['initial_state_sha256']==item['initialization']['initial_state_sha256']
 assert tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256'] and tr['signal_state_before_sha256']==tr['signal_state_after_sha256']
 assert tr['trainable_tensors']==tr['nonzero_gradient_tensors']==203 and not tr['missing_nonzero_gradients'] and tr['overflow_events']==0
 cache=collections.OrderedDict();warmup=65 if mode=='q1' else 2;seen=set();candidates=0;fresh=64;vjps=0;act=0;direct=[];zero_groups=0;elements=0;age_counts=collections.Counter();grad_changes={r:0 for r in ROLES};upstream_max_error=0.;mining_counts=collections.Counter();role_negative=collections.Counter()
 with (d/'memory_distances.f32').open('rb') as binary:
  for s,(row,lossrow) in enumerate(zip(rows,tr['steps'])):
   ix=row['record_indices']; seen.update(ix)
   assert row['step']==lossrow['step']==s+1 and row['zero_based_step']==s
   assert ix==lossrow['sampled_record_indices']==meta['folds'][fold['fold']]['batches'][0 if dirname.startswith('overfit') else s]['record_indices']
   assert set(ix)<=set(fold['source_record_indices']) and sorted(collections.Counter(records[i]['identity'] for i in ix).values())==[8]*8
   assert row['identities']==[records[i]['identity'] for i in ix] and row['scenes']==[records[i]['scene'] for i in ix]
   for i,t in list(cache.items()):
    if s-t>8: del cache[i]
   wanted=[dict(record_index=i,identity=records[i]['identity'],scene=records[i]['scene'],age=s-t,stored_step=t) for i,t in cache.items() if i not in ix]
   assert wanted==row['memory'];m=len(wanted);candidates+=m;act+=bool(m);age_counts.update(x['age'] for x in wanted)
   assert row['replacement_active']==(s>=warmup) and row['warmup_steps']==warmup
   assert row['current_anchor_count']==64 and row['history_anchor_count']==0 and row['coordinate_rule']=='fresh'
   assert row['history_candidate_vjp_applied']==(end=='history_gradient') and row['gradient_weight']==1
   assert binary.tell()==row['distance_offset_bytes'];num=64*(64+m)
   v=np.fromfile(binary,dtype='<f4',count=num);assert len(v)==num==row['distance_float_count'] and np.isfinite(v).all() and (v>=0).all()
   elements+=num;dc=v[:4096].reshape(64,64);dh=v[4096:].reshape(64,m)
   ids=np.array(row['identities']);scenes=np.array(row['scenes']);mids=np.array([x['identity'] for x in wanted]);msc=np.array([x['scene'] for x in wanted])
   same=ids[:,None]==ids[None,:];pos=same & ~np.eye(64,dtype=bool);neg=~same
   hp=np.max(np.where(pos,dc,-np.inf),axis=1);hn=np.min(np.where(neg,dc,np.inf),axis=1)
   mp=ids[:,None]==mids[None,:];mn=~mp
   mph=np.max(np.where(mp,dh,-np.inf),axis=1) if m else np.full(64,-np.inf)
   mnh=np.min(np.where(mn,dh,np.inf),axis=1) if m else np.full(64,np.inf)
   hpos=np.maximum(hp,mph);hneg=np.minimum(hn,mnh);hinge=hpos-hneg+np.float32(.3)
   basic=float(np.maximum(hp-hn+np.float32(.3),0).mean());expanded=float(np.maximum(hinge,0).mean())
   stats=dict(memory_records=m,memory_positive_pairs=int(mp.sum()),memory_negative_pairs=int(mn.sum()),memory_cross_scene_positive_pairs=int((mp&(scenes[:,None]!=msc)).sum()),memory_negative_violations_against_batch_hard_positive=int((mn&(dh<hp[:,None]+np.float32(.3))).sum()),harder_positive_anchors=int((mph>hp).sum()),harder_negative_anchors=int((mnh<hn).sum()),current_wrong_order_anchors=int((hp>=hn).sum()),expanded_wrong_order_anchors=int((hpos>=hneg).sum()),expanded_hinge_positive_anchors=int((hinge>0).sum()),maximum_memory_age=max([x['age'] for x in wanted],default=0))
   for k,val in stats.items():assert row['statistics'][k]==val,(dirname,s,k)
   mining_counts.update({k:v for k,v in stats.items() if k not in ('maximum_memory_age','memory_records')})
   for a,b in [(row['original_triplet'],basic),(row['statistics']['current_triplet'],basic),(row['statistics']['expanded_triplet'],expanded),(lossrow['components']['triplet_fused'],expanded if s>=warmup else basic)]:
    max_triplet=max(max_triplet,abs(a-b));close(a,b,2e-6)
   comps=lossrow['components'];assert len(comps)==14
   weight={'id_fused':.25,'triplet_fused':1.}
   weight.update({prefix+'_'+r:w for r in ROLES for prefix,w in [('id',1/12),('triplet',.25),('id_residual',1/12),('triplet_residual',.25)]})
   assert set(comps)==set(weight); calculated=math.fsum(comps[k]*weight[k] for k in comps)
   max_scalar=max(max_scalar,abs(calculated-lossrow['loss']));close(calculated,lossrow['loss'],1e-5)
   assert lossrow['amp_scale_after']>=lossrow['amp_scale_before']>0
   norms=np.array(row['historical_leaf_upstream_norms']);assert len(norms)==m and np.isfinite(norms).all() and (norms>=0).all()
   groups=sorted({x['stored_step'] for x,n in zip(wanted,norms) if n>0});available={x['stored_step'] for x in wanted}
   assert groups==row['history_vjp_groups'] and row['history_vjp_record_forwards']==64*len(groups)
   assert row['fresh_role_record_forwards']==64*len(available)
   fresh+=64*len(available);vjps+=64*len(groups);zero_groups+=len(available)-len(groups)
   # Reconstruct leaf-gradient norms geometrically from saved Euclidean distances.
   # This is a saved-coordinate check, not an independent model/VJP backward.
   if m:
    coeff=np.zeros((m,64),dtype=np.float64)
    for a in range(64):
     if hinge[a]<=0:continue
     if mph[a]>=hp[a] and np.isfinite(mph[a]):
      j=int(np.argmax(np.where(mp[a],dh[a],-np.inf)))
      if dh[a,j]>0:coeff[j,a]+=(.5 if mph[a]==hp[a] else 1.)/(64.*float(dh[a,j]))
     if mnh[a]<=hn[a] and np.isfinite(mnh[a]):
      j=int(np.argmin(np.where(mn[a],dh[a],np.inf)))
      if dh[a,j]>0:coeff[j,a]-=(.5 if mnh[a]==hn[a] else 1.)/(64.*float(dh[a,j]))
    computed=np.zeros(m)
    for j in np.flatnonzero(np.any(coeff!=0,axis=1)):
     indexes=np.flatnonzero(coeff[j]);c=coeff[j,indexes];squared=dh[indexes,j].astype(float)**2
     gram=(squared[:,None]+squared[None,:]-dc[np.ix_(indexes,indexes)].astype(float)**2)/2
     computed[j]=math.sqrt(max(0,float(c@gram@c)))
    error=float(np.max(np.abs(computed-norms)));upstream_max_error=max(upstream_max_error,error);max_upstream=max(max_upstream,error)
    assert error<2e-5,(dirname,s,error)
    assert not np.any((norms>0)&(~np.any(coeff!=0,axis=1)))
   for r in ROLES:
    a=row['roles'][r]['total_vs_history'];b=row['roles'][r]['total_vs_both'];ap=row['applied_gradients'][r]
    for v in (a,b,ap):
     x,y,z=v['first_norm'],v['second_norm'],v['difference_norm'];cos=v['cosine']
     assert all(math.isfinite(u) and u>=0 for u in (x,y,z))
     assert (cos is None)==(x==0 or y==0)
     if cos is not None:
      assert abs(cos)<=1.00001;close(z*z,x*x+y*y-2*x*y*cos,1e-7*max(1,x*x+y*y))
    close(a['first_norm'],b['first_norm'],1e-8);close(b['difference_norm'],a['second_norm'],1e-6*max(1,a['second_norm']))
    if end=='history_gradient':assert ap==b
    else:assert ap['first_norm']==ap['second_norm']==a['first_norm'] and ap['difference_norm']==0
    grad_changes[r]+=ap['difference_norm']>0
    role_negative[r]+=a['cosine'] is not None and a['cosine']<0
   for flag in ('selected_reencoding_bitwise','history_rng_buffers_preserved','history_vjp_leaves_current_grad_unchanged','final_gradient_addition_bitwise'):assert row[flag]
   if row['direct_single_group_check']:
    proof=row['direct_single_group_check'];assert len(available)==1
    close(proof['relative_l2_error'],proof['difference_norm']/proof['first_norm'],1e-12);assert proof['relative_l2_error']<=.005
    direct.append(dict(step=s+1,relative_l2_error=proof['relative_l2_error']))
   if s>=warmup:
    for i in ix:cache.pop(i,None);cache[i]=s
    while len(cache)>512:cache.popitem(last=False)
  assert not binary.read()
 assert fresh==tr['extra_fresh_role_record_forwards'] and vjps==tr['extra_history_vjp_record_forwards']
 assert len(direct)*64==tr['extra_direct_check_record_forwards']
 assert len(direct)==(1 if mode=='m0' and not dirname.startswith('overfit') else 0)
 if mode=='q1':assert seen==set(fold['source_record_indices'])
 for epoch in tr['history']:
  selected=[x for x in tr['steps'] if x['epoch']==epoch['epoch']]
  assert len(selected)==epoch['optimizer_steps'];close(mean([x['loss'] for x in selected]),epoch['mean_loss'],1e-12)
  e=epoch['epoch'];factor=e/5 if e<=5 else .5*(1+math.cos(math.pi*(e-6)/15))
  close(epoch['learning_rate'],.00035*(factor if mode=='q1' else 1),1e-15)
 PAIRS[mode,fold['fold'],end,dirname.startswith('overfit')]=[(x['record_indices'],x['pixel_sha256'],x['memory']) for x in rows]
 result=dict(mode=mode,fold=fold['fold'],endpoint=end,directory=dirname,updates=count,seen_source_records=len(seen),historical_updates=act,candidate_record_exposures=candidates,ages=dict(age_counts),fresh_record_forwards=fresh,vjp_record_forwards=vjps,zero_upstream_group_skips=zero_groups,distance_elements=elements,leaf_norm_max_absolute_error=upstream_max_error,applied_changed_steps_by_role=grad_changes,total_vs_history_negative_cosines=role_negative,mining_exposures=dict(mining_counts),direct_single_group_checks=direct,training_seconds=sum(x['elapsed_seconds'] for x in tr['history']),peak_allocated_mib=tr['peak_allocated_mib'],peak_reserved_mib=tr['peak_reserved_mib'])
 total_distance+=elements;training_results.append(result)
 if mode=='q1':q1_rows[fold['fold'],end]=rows

checkpoint_results=[];retrieval_results=[];ap_by={end:{o:[] for o in OUTPUTS} for end in ENDS};first_by={end:{o:[] for o in OUTPUTS} for end in ENDS};foldmetrics={end:[] for end in ENDS};query_records={end:{o:[] for o in OUTPUTS} for end in ENDS};rank_positions=0
for f,fold in enumerate(protocol['folds']):
 b0=baseline['folds'][f];original=tensor_file(b0['checkpoint']);assert HASHES[b0['checkpoint']]['sha256']==b0['checkpoint_sha256']
 assert original['source_ids']==fold['source_ids'] and original['heldout_ids']==fold['heldout_ids']
 assert state_hash(original['model_state_dict'])==b0['training']['final_state_sha256']
 ref=tensor_file(Path(b0['checkpoint']).parent/'retrieval_arrays.pt')
 assert HASHES[str(Path(b0['checkpoint']).parent/'retrieval_arrays.pt')]['sha256']==b0['retrieval']['retrieval_arrays_sha256']
 for mode,source_summary in [('m0',m0),('q1',summary)]:
  for end in ENDS:
   item=source_summary['folds'][f]['endpoints'][end];dirname=f'fold_{f}_{end}';directory=RUN/mode/dirname
   assert read(directory/'receipt.json')==item;training(mode,fold,end,item,dirname)
   assert item['initialization']==m0['folds'][f]['endpoints'][end]['initialization']
   ck=tensor_file(item['checkpoint']);assert HASHES[item['checkpoint']]['sha256']==item['checkpoint_sha256']
   assert ck['binding']==item['initialization'] and ck['config_sha256']==summary['config_sha256']
   assert ck['source_ids']==fold['source_ids'] and ck['heldout_ids']==fold['heldout_ids'] and ck['fold']==f
   combined={k:original['model_state_dict'][v] for k,v in ck['baseline_aliases'].items()};combined.update(ck['role_state_dict'])
   final=state_hash(combined);assert final==item['training']['final_state_sha256']==item['strict_reload_state_sha256']
   assert len([n for n in item['initialization']['trainable_names'] if n.startswith('encoder.')])==189
   checkpoint_results.append(dict(mode=mode,fold=f,endpoint=end,checkpoint=item['checkpoint'],sha256=item['checkpoint_sha256'],reconstructed_state_sha256=final,encoder_trainable_tensors=189,all_trainable_tensors=203))
   del combined,ck
   if mode=='m0':continue
   saved=item['retrieval'];ranks=read(directory/'rankings.json');arrays=tensor_file(directory/'retrieval_arrays.pt')
   assert HASHES[str(directory/'rankings.json')]['sha256']==saved['rankings_sha256']
   assert HASHES[str(directory/'retrieval_arrays.pt')]['sha256']==saved['retrieval_arrays_sha256']
   gal=[records[i] for i in fold['gallery_record_indices']];positions=[q['gallery_position'] for q in fold['query_rows']]
   assert saved['gallery_manifest']==gal and saved['query_rows']==fold['query_rows']
   assert arrays['gallery_record_indices']==fold['gallery_record_indices'] and arrays['query_gallery_positions']==positions
   assert torch.equal(arrays['features']['baseline_only'],ref['features']) and torch.equal(arrays['distances']['baseline_only'],ref['distances'])
   assert torch.equal(arrays['features']['fused'][:,:3072],ref['features'])
   metrics={};torch.set_num_threads(56)
   for o,width in zip(OUTPUTS,(3072,7680,4608,4608,4608)):
    x=arrays['features'][o].float();dist=arrays['distances'][o];assert x.shape==(len(gal),width) and torch.isfinite(x).all()
    unit=x/torch.linalg.vector_norm(x,dim=1,keepdim=True).clamp_min(1e-12);q=unit[positions]
    calc=(q*q).sum(1)[:,None]+(unit*unit).sum(1)[None,:];calc.addmm_(q,unit.T,beta=1,alpha=-2)
    assert torch.equal(calc,dist),(f,end,o,'distance mismatch')
    order=np.argsort(dist.numpy(),axis=1);assert order.tolist()==ranks[o]
    aps=[];firsts=[];tie_queries=0;mixed_tie_queries=0
    for qi,(p,seq) in enumerate(zip(positions,order.tolist())):
     assert sorted(seq)==list(range(len(gal)));query=gal[p]
     legal=[j for j in seq if gal[j]['identity']!=query['identity'] or gal[j]['scene']!=query['scene']]
     hit=[k+1 for k,j in enumerate(legal) if gal[j]['identity']==query['identity']]
     assert len(hit)==fold['query_rows'][qi]['valid_positives']
     ap=math.fsum((n+1)/rank for n,rank in enumerate(hit))/len(hit);first=hit[0]
     close(ap,saved['outputs'][o]['average_precision'][qi],1e-14);assert first==saved['outputs'][o]['first_match_rank'][qi]
     aps.append(ap);firsts.append(first);neg=next(j for j in legal if gal[j]['identity']!=query['identity'])
     query_records[end][o].append(dict(fold=f,record_index=query['index'],identity=query['identity'],scene=query['scene'],ap=ap,first_match_rank=first,last_positive_rank=hit[-1],positive_count=len(hit),nearest_negative_record=gal[neg]['index'],nearest_negative_identity=gal[neg]['identity'],nearest_negative_scene=gal[neg]['scene']))
     vals=dist[qi,legal].numpy();equal=vals[1:]==vals[:-1];tie_queries+=bool(equal.any());mixed_tie_queries+=any(equal[k] and (gal[legal[k]]['identity']==query['identity'])!=(gal[legal[k+1]]['identity']==query['identity']) for k in range(len(equal)))
    score=dict(mAP=mean(aps)*100,**{f'Rank-{k}':sum(v<=k for v in firsts)*100/len(firsts) for k in (1,5,10)})
    for k,v in score.items():close(v,saved['outputs'][o]['metrics'][k])
    metrics[o]=score;ap_by[end][o].extend(aps);first_by[end][o].extend(firsts);rank_positions+=order.size
    retrieval_results.append(dict(fold=f,endpoint=end,output=o,metrics=score,distance_elements=int(order.size),exact_saved_distance_and_rank_replay=True,legal_distance_tie_queries=tie_queries,mixed_relevance_tie_queries=mixed_tie_queries))
   foldmetrics[end].append(metrics)
 for mode in ('m0','q1'):
  assert PAIRS[mode,f,'control',False]==PAIRS[mode,f,'history_gradient',False]
  s=m0 if mode=='m0' else summary;assert s['folds'][f]['endpoints']['control']['initialization']==s['folds'][f]['endpoints']['history_gradient']['initialization']
for end in ENDS:training('m0',protocol['folds'][0],end,m0['overfit'][end],'overfit_'+end)
assert PAIRS['m0',0,'control',True]==PAIRS['m0',0,'history_gradient',True]

ids=np.array([q['identity'] for f in protocol['folds'] for q in f['query_rows']]);unique=np.unique(ids)
def lower(delta):
 means=np.array([mean(delta[ids==i]) for i in unique]);sizes=np.array([(ids==i).sum() for i in unique]);draws=np.random.default_rng(42).integers(len(unique),size=(10000,len(unique)))
 values=(means[draws]*sizes[draws]).sum(axis=1)/sizes[draws].sum(axis=1)
 return float(np.quantile(values,.025,method='linear'))
ap_by={e:{o:np.array(v) for o,v in d.items()} for e,d in ap_by.items()};first_by={e:{o:np.array(v) for o,v in d.items()} for e,d in first_by.items()}
endpoint_results={}
for end in ENDS:
 saved=summary['comparison']['endpoints'][end]
 metrics={o:dict(mAP=mean(ap_by[end][o])*100,**{f'Rank-{k}':float(np.mean(first_by[end][o]<=k)*100) for k in (1,5,10)}) for o in OUTPUTS}
 for o in OUTPUTS:
  for k,v in metrics[o].items():close(v,saved['metrics'][o][k])
 gains={o:metrics[o]['mAP']-metrics['baseline_only']['mAP'] for o in OUTPUTS};foldg=[r['fused']['mAP']-r['baseline_only']['mAP'] for r in foldmetrics[end]];boot=lower((ap_by[end]['fused']-ap_by[end]['baseline_only'])*100)
 checks=dict(fused_gain_at_least_1pp=gains['fused']>=1,all_fold_fused_gains_nonnegative=all(v>=0 for v in foldg),all_full_branches_not_below_signal=all(gains[r]>=0 for r in ROLES),identity_bootstrap_lower_positive=boot>0,fused_strictly_best=all(metrics['fused']['mAP']>metrics[o]['mAP'] for o in OUTPUTS if o!='fused'))
 assert checks==saved['scientific_checks'];close(boot,saved['identity_bootstrap']['lower_bound_pp'])
 for i in saved['per_identity']:
  mask=ids==i['identity'];assert sum(mask)==i['query_count']
  for o in OUTPUTS:close(mean(ap_by[end][o][mask])*100,i['map_by_output'][o])
 for o in ROLES+('fused',):
  delta=ap_by[end][o]-ap_by[end]['baseline_only'];f=first_by[end][o];b=first_by[end]['baseline_only']
  changes=dict(ap_improved=int(sum(delta>0)),ap_declined=int(sum(delta<0)),ap_unchanged=int(sum(delta==0)),rank1_repaired=int(sum((b>1)&(f==1))),rank1_new_errors=int(sum((b==1)&(f>1))))
  assert changes==saved['query_changes'][o]
 endpoint_results[end]=dict(metrics=metrics,gains=gains,fold_fused_gains=foldg,bootstrap_lower_pp=boot,checks=checks)
comparison=summary['comparison'];deltas={o:(ap_by['history_gradient'][o]-ap_by['control'][o])*100 for o in OUTPUTS};gains={o:mean(v) for o,v in deltas.items()};foldg=[b['fused']['mAP']-a['fused']['mAP'] for a,b in zip(foldmetrics['control'],foldmetrics['history_gradient'])];boot=lower(deltas['fused'])
checks=dict(fused_gain_at_least_1pp=gains['fused']>=1,all_fold_fused_nonnegative=all(v>=0 for v in foldg),all_role_gains_nonnegative=all(gains[r]>=0 for r in ROLES),paired_identity_bootstrap_lower_positive=boot>0,candidate_fused_strictly_best=endpoint_results['history_gradient']['checks']['fused_strictly_best'])
assert checks==comparison['paired_checks'];close(boot,comparison['paired_bootstrap_lower_pp'])
assert all(checks.values())==comparison['paired_pass'] and all(endpoint_results['history_gradient']['checks'].values())==comparison['vehicle_baseline_pass']
assert comparison['next_phase_qualified']==(all(checks.values()) and comparison['vehicle_baseline_pass'])
assert summary['status']==('Q1_PASS' if comparison['next_phase_qualified'] else 'Q1_FAIL')
changes={};identities_out=[]
for o in OUTPUTS:
 close(gains[o],comparison['matched_gains_mAP'][o]);delta=deltas[o];a=first_by['control'][o];b=first_by['history_gradient'][o]
 changes[o]=dict(ap_improved=int(sum(delta>0)),ap_declined=int(sum(delta<0)),ap_unchanged=int(sum(delta==0)),rank1_repaired=int(sum((a>1)&(b==1))),rank1_new_errors=int(sum((a==1)&(b>1))))
 for i in unique:
  mask=ids==i;gain=mean(delta[mask]);record=next(x for x in comparison['paired_per_identity'] if x['identity']==i);close(gain,record['gains_mAP'][o]);assert record['query_count']==sum(mask)
  identities_out.append(dict(identity=int(i),output=o,query_count=int(sum(mask)),delta_mAP_pp=gain,control_mAP=mean(ap_by['control'][o][mask])*100,candidate_mAP=mean(ap_by['history_gradient'][o][mask])*100))
paired=dict(gains=gains,fold_fused_gains=foldg,bootstrap_lower_pp=boot,checks=checks,changes=changes,identity_changes={o:dict(improved=sum(r['output']==o and r['delta_mAP_pp']>0 for r in identities_out),declined=sum(r['output']==o and r['delta_mAP_pp']<0 for r in identities_out),unchanged=sum(r['output']==o and r['delta_mAP_pp']==0 for r in identities_out)) for o in OUTPUTS})
for mode,proof in [('q1',cpu),('m0',m0cpu)]:
 items=[x for x in training_results if x['mode']==mode]
 assert sum(x['updates'] for x in items)==proof['checked_training_steps']
 assert sum(x['distance_elements'] for x in items)==proof['checked_memory_distance_elements']
 assert sum(x['vjp_record_forwards'] for x in items)==proof['checked_history_vjp_record_forwards']
assert rank_positions==cpu['checked_retrieval_distance_and_rank_elements']==2069520
result=dict(status='PASS_INDEPENDENT_SAVED_ARTIFACT_REPLAY',scientific_status=summary['status'],protocol=protocol_result,training=training_results,checkpoints=checkpoint_results,retrieval=retrieval_results,endpoints=endpoint_results,paired=paired,query_records=query_records,identity_records=identities_out,max_weighted_scalar_error=max_scalar,max_fused_triplet_error=max_triplet,max_leaf_norm_distance_geometry_error=max_upstream,training_distance_elements=total_distance,ranking_distance_elements=rank_positions,audited_input_hashes=HASHES,model_forwards=0,optimizer_updates=0,official_image_reads=0,torch_version=torch.__version__,numpy_version=np.__version__,elapsed_seconds=time.monotonic()-START)
print(json.dumps(result,indent=2))
