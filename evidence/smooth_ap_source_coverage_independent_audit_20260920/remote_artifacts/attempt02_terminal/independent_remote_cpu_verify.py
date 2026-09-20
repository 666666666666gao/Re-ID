"""Independent CPU audit; no imports from project or executor verifier; no writes remotely."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[key]='4'
from pathlib import Path
from collections import Counter,OrderedDict
import json,hashlib,datetime,time,re,resource
import numpy as np
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590')
started=time.perf_counter()
def emit(kind,**values):
 print(json.dumps(dict(kind=kind,**values),ensure_ascii=False),flush=True)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def jload(p):return json.loads(Path(p).read_bytes())
emit('start',pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),numpy=np.__version__,threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
spec=jload(repo/'configs/MSVR310/TriFusion-smooth-ap-source-coverage-v1.json')
protocol=jload(repo/spec['protocol'])
summary=jload(root/'summary.json')
coverage=jload(root/'coverage.json')
q1=jload(spec['q1_summary'])
assert digest(spec['q1_summary'])==spec['q1_summary_sha256']
assert summary['status']=='COMPLETE_SOURCE_EXTRACTION'
assert coverage['status']=='COMPLETE_SOURCE_COVERAGE'
assert summary['contract_sha256']==digest(repo/'configs/MSVR310/TriFusion-smooth-ap-source-coverage-v1.json')
expected_conditions=[f'fold_{f}_{e}_{v}' for f in range(3) for e in ('control','smooth_ap') for v in ('clean','augmented')]
assert [c['directory'] for c in summary['conditions']]==expected_conditions
assert [c['directory'] for c in coverage['conditions']]==expected_conditions
assert summary['source_record_forwards']==8256 and summary['optimizer_updates']==0
assert summary['heldout_record_forwards']==summary['official_image_reads']==0
records=protocol['records']
assert [r['index'] for r in records]==list(range(1032))
assert len({tuple(r['paths']) for r in records})==1032
for r in records:
 names=[Path(p).name for p in r['paths']]
 assert len(set(names))==1
 match=re.fullmatch(r'(\d+)_s(\d+)_v(\d+)_(\d+)\.jpg',names[0]);assert match,names
 pid,scene,camera,_=map(int,match.groups())
 assert (pid,scene,camera)==(r['identity'],r['scene'],r['camera'])
 assert [Path(p).parts[-2] for p in r['paths']]==['vis','ni','th']
 assert all(Path(p).parts[0]=='bounding_box_train' and int(Path(p).parts[1])==pid for p in r['paths'])
membership=Counter()
fold_info=[]
for fold in protocol['folds']:
 indices=fold['source_record_indices'];source=set(fold['source_ids']);held=set(fold['heldout_ids'])
 assert indices==sorted(set(indices)) and not source&held
 assert source=={records[i]['identity'] for i in indices}
 assert indices==[i for i,r in enumerate(records) if r['identity'] in source]
 assert source|held=={r['identity'] for r in records}
 assert fold['gallery_record_indices']==[i for i,r in enumerate(records) if r['identity'] in held]
 membership.update(indices)
 labels=np.array([records[i]['identity'] for i in indices]);scenes=np.array([records[i]['scene'] for i in indices])
 all_n=[sum(labels==labels[i])-1 for i in range(len(indices))]
 cross_n=[sum((labels==labels[i])&(scenes!=scenes[i])) for i in range(len(indices))]
 fold_info.append(dict(fold=fold['fold'],source_records=len(indices),identities=len(source),scenes=len(set(scenes.tolist())),all_identity_eligible=int(sum(x>0 for x in all_n)),cross_scene_eligible=int(sum(x>0 for x in cross_n)),single_scene_records=int(sum(x==0 for x in cross_n))))
assert set(membership)==set(range(1032)) and set(membership.values())=={2}
emit('protocol',sha256=digest(repo/spec['protocol']),records=1032,identities=len({r['identity'] for r in records}),each_record_source_membership=2,folds=fold_info,dataset_filename_ground_truth_consistent=True)
logs_by_key={}
for fold in protocol['folds']:
 for end in ('control','smooth_ap'):
  qr=q1['folds'][fold['fold']]['endpoints'][end]
  path=Path(qr['checkpoint']).parent/'memory_steps.jsonl'; rows=[json.loads(l) for l in path.read_text().splitlines()]
  assert digest(path)==qr['training']['audit_files']['memory_steps.jsonl']['sha256']
  assert len(rows)==260 and [r['step'] for r in rows]==list(range(1,261))
  mem=OrderedDict();seen=Counter();duplicates=0;pools=[]
  for step,row in enumerate(rows):
   inds=row['record_indices']; seen.update(inds);duplicates+=len(inds)-len(set(inds))
   assert len(inds)==64 and set(inds)<=set(fold['source_record_indices'])
   assert row['identities']==[records[i]['identity'] for i in inds]
   assert row['scenes']==[records[i]['scene'] for i in inds]
   assert sorted(Counter(row['identities']).values())==[8]*8
   for i in list(mem):
    if step-mem[i]>8:del mem[i]
   expected=[dict(record_index=i,identity=records[i]['identity'],scene=records[i]['scene'],age=step-s,stored_step=s) for i,s in mem.items() if i not in set(inds)]
   assert row['memory']==expected,(fold['fold'],end,step)
   pool=sorted(set(inds)|{m['record_index'] for m in row['memory']});pools.append(pool)
   assert set(pool)<=set(fold['source_record_indices'])
   assert row['warmup_steps']==65 and row['replacement_active']==(step>=65)
   if step>=65:
    for i in inds:
     mem.pop(i,None);mem[i]=step
    while len(mem)>512:mem.popitem(last=False)
  logs_by_key[(fold['fold'],end)]=(rows,pools)
  emit('memory_pool_inputs',fold=fold['fold'],endpoint=end,sha256=digest(path),rows=len(rows),source_records_seen=len(seen),unseen_source_records=sorted(set(fold['source_record_indices'])-set(seen)),anchor_positions=sum(seen.values()),repeated_positions_within_batches=duplicates,pool_min=min(map(len,pools)),pool_max=max(map(len,pools)),ordered_memory_reconstruction_exact=True)
 for a,b in zip(logs_by_key[(fold['fold'],'control')][0],logs_by_key[(fold['fold'],'smooth_ap')][0]):
  assert a['record_indices']==b['record_indices'] and a['memory']==b['memory'] and a['pixel_sha256']==b['pixel_sha256']
widths={'baseline_only':3072,'fused':7680,'cnn':4608,'transformer':4608,'mamba':4608}
def measure(distance,order,labels,scenes,q,present,cross):
 legal=present.copy(); legal[q]=False
 if cross:legal&=~((labels==labels[q])&(scenes==scenes[q]))
 ordered=order[legal[order]]
 hit=(labels[ordered]==labels[q])
 p=int(np.count_nonzero(hit));n=len(ordered)-p
 if not p:return dict(eligible=False,positives=0,negatives=n,ap=None,rank1=None,inverted_positives=None)
 ranks=np.flatnonzero(hit)+1
 numerator=np.cumsum(hit,dtype=np.int64)[hit]
 ap=float(np.sum(numerator/ranks,dtype=np.float64)/p)
 nearest=distance[ordered[~hit]].min(initial=np.inf)
 return dict(eligible=True,positives=p,negatives=n,ap=ap,rank1=int(hit[0]),inverted_positives=int(np.count_nonzero(distance[ordered[hit]]>nearest)))
max_ap_error=0.;full_rows=0;candidate_rows=0;array_rows=0;out_rows=[];max_distance_error=0.
def same(actual,expected):
 global max_ap_error
 assert set(actual)==set(expected),(set(actual),set(expected))
 for k,v in actual.items():
  if k=='ap' and v is not None:
   err=abs(v-expected[k]);max_ap_error=max(max_ap_error,err);assert err<=1e-14,(actual,expected)
  else:assert v==expected[k],(k,v,expected[k])
# Semantic adversarial fixtures include both exact ties and records without a legal query positive.
lab=np.array([0,0,0,1,2]);sc=np.array([0,0,1,0,0]);dv=np.array([0.,.1,.4,.2,.8]);pr=np.ones(5,dtype=bool);order=np.lexsort((np.arange(5),dv))
assert measure(dv,order,lab,sc,0,pr,True)['ap']==.5
assert abs(measure(dv,order,lab,sc,0,pr,False)['ap']-5/6)<1e-15
pr2=pr.copy();pr2[2]=False
assert measure(dv,order,lab,sc,0,pr2,True)==dict(eligible=False,positives=0,negatives=2,ap=None,rank1=None,inverted_positives=None)
assert measure(dv,order,lab,sc,3,pr,True)['negatives']==4
tie=np.array([0.,.2,.2]);ids=np.array([0,1,0]);env=np.zeros(3,dtype=int)
assert measure(tie,np.lexsort((np.arange(3),tie)),ids,env,0,np.ones(3,dtype=bool),False)['ap']==.5
emit('semantic_fixtures',status='PASS',self=True,multi_positive_denominator=True,same_identity_same_scene_removed=True,no_positive_query_retained_as_negative=True,global_index_exact_tie=True)
for cond in summary['conditions']:
 cstart=time.perf_counter();name=cond['directory'];directory=root/name
 indices=cond['record_indices'];fold=protocol['folds'][cond['fold']]
 assert indices==fold['source_record_indices']
 receipt=jload(directory/'receipt.json')
 assert {k:v for k,v in cond.items() if k!='directory'}==receipt
 assert cond['state_sha256']==q1['folds'][cond['fold']]['endpoints'][cond['endpoint']]['training']['final_state_sha256']
 assert all(cond[k] for k in ('model_state_unchanged','gradients_absent','paired_pixels_exact','paired_baseline_exact'))
 labels=np.array([records[i]['identity'] for i in indices]);scenes=np.array([records[i]['scene'] for i in indices]);idx=np.array(indices);lookup={v:i for i,v in enumerate(indices)}
 n=len(indices);all_present=np.ones(n,dtype=bool)
 ip=directory/'inputs.jsonl'; inputs=[json.loads(l) for l in ip.read_text().splitlines()]
 assert [r['batch'] for r in inputs]==list(range((n+63)//64))
 assert [i for r in inputs for i in r['record_indices']]==indices
 assert [r['record_indices'] for r in inputs]==[indices[k:k+64] for k in range(0,n,64)]
 ctrl=root/f'fold_{cond["fold"]}_control_{cond["view"]}'
 assert inputs==[json.loads(l) for l in (ctrl/'inputs.jsonl').read_text().splitlines()]
 for fn,entry in cond['files'].items():
  p=directory/fn;assert digest(p)==entry['sha256'] and p.stat().st_size==entry['bytes']
 assert set(cond['files'])=={'inputs.jsonl',*[k+'.npy' for k in widths]}
 assert sorted(p.stem for p in directory.glob('*.npy'))==sorted(widths)
 batches,pools=logs_by_key[(cond['fold'],cond['endpoint'])]
 pool_masks=[]
 for pool in pools:
  mask=np.zeros(n,dtype=bool);mask[[lookup[i] for i in pool]]=True;pool_masks.append(mask)
 for output,width in widths.items():
  emit('array_start',condition=name,output=output,elapsed_seconds=time.perf_counter()-started)
  path=directory/(output+'.npy');a=np.load(path,allow_pickle=False)
  assert a.shape==(n,width) and a.dtype==np.float32 and np.isfinite(a).all()
  norms=np.linalg.norm(a.astype(np.float64),axis=1);norm_error=float(np.abs(norms-1).max());assert norm_error<2e-6 and min(norms)>0
  if output=='baseline_only':assert np.array_equal(a,np.load(ctrl/(output+'.npy'),allow_pickle=False))
  x=a.astype(np.float64);del a;array_rows+=n
  # The registered algebraic Float64 distances control ties; direct differences independently check geometry.
  squared=np.einsum('ij,ij->i',x,x)
  registered=np.maximum((x*x).sum(1)[:,None]+(x*x).sum(1)[None,:]-2*x@x.T,0)
  direct=np.empty((n,n))
  for i in range(n):
   diff=x-x[i];direct[i]=np.sum(diff*diff,axis=1)
   if i and i%256==0:emit('geometry_progress',condition=name,output=output,queries=i)
  derr=float(np.max(np.abs(registered-direct)));assert derr<1e-12;max_distance_error=max(max_distance_error,derr)
  orders=np.stack([np.lexsort((idx,registered[i])) for i in range(n)])
  direct_orders=np.stack([np.lexsort((idx,direct[i])) for i in range(n)])
  tie_pairs=0;cross_identity_ties=0
  for i,ordr in enumerate(orders):
   nonself=ordr[ordr!=i];equal=registered[i,nonself[1:]]==registered[i,nonself[:-1]]
   tie_pairs+=int(equal.sum());cross_identity_ties+=int(np.sum(equal&(labels[nonself[1:]]!=labels[nonself[:-1]])))
  emit('array_geometry',condition=name,output=output,shape=list(x.shape),sha256=digest(path),norm_max_error=norm_error,distance_max_error=derr,exact_nonself_adjacent_distance_ties=tie_pairs,cross_identity_ties=cross_identity_ties,direct_vs_registered_order_changed_queries=int(np.any(orders!=direct_orders,axis=1).sum()))
  for cross in (False,True):
   stem=output+('_cross_scene' if cross else '_all_identity')
   saved_full=jload(directory/(stem+'_full.json'));assert len(saved_full)==n
   expected_full=[];direct_ap_diff=0.;direct_ap_changed=0
   for i,row in enumerate(saved_full):
    assert row['record_index']==indices[i]
    measured=measure(registered[i],orders[i],labels,scenes,i,all_present,cross)
    same(measured,{k:v for k,v in row.items() if k!='record_index'})
    expected_full.append(measured);full_rows+=1
    direct_row=measure(direct[i],direct_orders[i],labels,scenes,i,all_present,cross)
    if measured['eligible']:
     err=abs(measured['ap']-direct_row['ap']);direct_ap_diff=max(direct_ap_diff,err);direct_ap_changed+=int(err>1e-14)
   pairs=[];missing=0;neither=0;cache={};source_exposure=Counter();rows=0;chash=hashlib.sha256()
   cp=directory/(stem+'_candidates.jsonl')
   with cp.open('rb') as stream:
    for raw in stream:
     chash.update(raw);row=json.loads(raw);step,pos=divmod(rows,64)
     assert step<260 and row['step']==step+1 and row['anchor_position']==pos
     record=batches[step]['record_indices'][pos];assert row['record_index']==record
     i=lookup[record];source_exposure[record]+=1;key=(step,i)
     if key not in cache:cache[key]=measure(registered[i],orders[i],labels,scenes,i,pool_masks[step],cross)
     same(cache[key],row['candidate']);same(expected_full[i],row['full'])
     if row['candidate']['eligible']:
      assert row['full']['eligible'];pairs.append((row['candidate']['ap'],row['full']['ap']))
     elif row['full']['eligible']:missing+=1
     else:neither+=1
     rows+=1;candidate_rows+=1
   assert rows==16640
   eligible=[r for r in expected_full if r['eligible']];pair_arr=np.array(pairs,dtype=np.float64)
   computed=dict(output=output,cross_scene=cross,full_eligible=len(eligible),full_source_map=float(100*np.mean([r['ap'] for r in eligible])),anchor_exposures=rows,common_eligible=len(pairs),pool_missing_positive=missing,common_candidate_map=float(100*pair_arr[:,0].mean()),common_full_map=float(100*pair_arr[:,1].mean()))
   saved=next(m for c in coverage['conditions'] if c['directory']==name for m in c['metrics'] if m['output']==output and m['cross_scene']==cross)
   assert set(saved)==set(computed)
   for key,v in computed.items():
    if isinstance(v,float):assert abs(saved[key]-v)<1e-10,(key,v,saved[key])
    else:assert saved[key]==v,(key,v,saved[key])
   row=dict(condition=name,**computed,source_rank1=float(100*np.mean([r['rank1'] for r in eligible])),inverted_positives=sum(r['inverted_positives'] for r in eligible),neither_eligible_exposures=neither,candidate_file_sha256=chash.hexdigest(),candidate_file_bytes=cp.stat().st_size,full_file_sha256=digest(directory/(stem+'_full.json')),direct_formula_full_ap_changed_queries=direct_ap_changed,direct_formula_max_ap_delta=direct_ap_diff)
   out_rows.append(row);emit('condition_check',**row)
  del x,registered,direct
 emit('condition_complete',condition=name,elapsed_seconds=time.perf_counter()-cstart)
assert len(out_rows)==120 and full_rows==82560 and candidate_rows==1996800 and array_rows==41280
emit('complete',status='PASS_INDEPENDENT_ALL_ARRAYS_AND_ROWS',conditions=12,output_filter_conditions=len(out_rows),arrays=60,array_rows=array_rows,full_rows=full_rows,candidate_rows=candidate_rows,max_ap_error=max_ap_error,max_distance_error=max_distance_error,model_forwards=0,optimizer_updates=0,torch_imported=False,cuda_initialized=False,image_reads=0,remote_writes=0,elapsed_seconds=time.perf_counter()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,utime=resource.getrusage(resource.RUSAGE_SELF).ru_utime,stime=resource.getrusage(resource.RUSAGE_SELF).ru_stime)

