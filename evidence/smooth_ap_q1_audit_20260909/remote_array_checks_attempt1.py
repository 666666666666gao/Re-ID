"""Fresh CPU-only auditor check. All arrays/checkpoints remain on the remote host.
No project model is constructed and no dataset image is opened.
"""
import os,sys,json,hashlib,time,math,datetime,subprocess,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
ROOT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4')
def load(p):return json.loads(p.read_bytes())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def state_sha(state):
 h=hashlib.sha256()
 for n,t in sorted(state.items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
started=time.perf_counter();torch.set_num_threads(1)
summary=load(RUN/'q1/summary.json');protocol=load(ROOT/'protocols/msvr310_train_oof_v1.json')
cfg=load(ROOT/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
base=load(ROOT/cfg['BASELINE']['CONFIG']);baseline=load(Path(cfg['BASELINE']['SUMMARY']))
out=dict(status='PASS',started_at=datetime.datetime.now().astimezone().isoformat(),scope='Q1 only: complete saved matrices, derivatives with respect to distance entries, retrieval features/distances/rankings, checkpoint tensors, source filename labels and bound code. No image/model/optimizer replay.',files={},source_checks=[],endpoints=[],model_forwards=0,optimizer_updates=0,image_reads=0,official_test_access=0)
def bind(p,expected=None):
 p=Path(p);actual=sha(p)
 if expected is not None:assert actual==expected,(str(p),actual,expected)
 out['files'][str(p)]=dict(bytes=p.stat().st_size,sha256=actual)
 return actual
bind(RUN/'q1/summary.json');bind(ROOT/'protocols/msvr310_train_oof_v1.json',base['protocol_sha256'])
bind(base['clip_weight'],base['clip_weight_sha256'])
for key,parent in [('project_source_file_sha256',ROOT),('signal_source_file_sha256',Path(base['signal_source']))]:
 for name,digest in base[key].items():bind(parent/name,digest)
out['comparator_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base['signal_source'],text=True).strip();assert out['comparator_commit']==base['signal_commit']
out['comparator_diff_sha256']=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=base['signal_source'])).hexdigest();assert out['comparator_diff_sha256']==base['signal_diff_sha256']
seen=[]
for r in protocol['records']:
 for relative in r['paths']:
  p=Path(base['dataset_root'])/relative;assert p.is_file() and p.stat().st_size>0;seen.append(relative)
actual_paths=sorted(str(p.relative_to(base['dataset_root'])) for p in (Path(base['dataset_root'])/'bounding_box_train').glob('*/*/*.jpg'))
assert sorted(seen)==actual_paths
out['source_path_inventory']=dict(protocol_paths=len(seen),actual_paths=len(actual_paths),all_train_files_present=True,official_directories_enumerated=False,image_contents_verified=False)
objpath=ROOT/'tools/msvr_smooth_ap.py';bind(objpath,'9d7c01830493233183b2cc9366ec8742caab1f782208870badc8dc93a94b3b68')
spec=importlib.util.spec_from_file_location('audited_objective_only',objpath);objective=importlib.util.module_from_spec(spec);spec.loader.exec_module(objective)
def independent_ap_derivative(d,ids,mids):
 """Closed-form chain rule in NumPy; no imported training objective."""
 B=len(ids);cand=np.array(list(ids)+list(mids));s=1-d.astype(np.float64)**2/2;g=np.zeros_like(s);aps=[]
 for i in range(B):
  pos=np.flatnonzero((cand==ids[i])&(np.arange(len(cand))!=i));positive=(cand==ids[i]);positive[i]=False
  z=(s[i][None,:]-s[i,pos][:,None])/.01;t=1/(1+np.exp(-z));valid=(np.arange(len(cand))[None,:]!=i)&(np.arange(len(cand))[None,:]!=pos[:,None]);t[~valid]=0
  a=1+t[:,positive].sum(1);b=1+t.sum(1);aps.append(float(np.mean(a/b)))
  coeff=-(positive[None,:]*b[:,None]-a[:,None])/(b[:,None]**2*len(pos)*B)
  partial=coeff*t*(1-t)/.01;partial[~valid]=0
  g[i]=partial.sum(0);g[i,pos]-=partial.sum(1)
 return np.array(aps),-g*d
def independent_hard_derivative(d,ids,mids):
 B=len(ids);g=np.zeros_like(d,dtype=np.float64);v=[];basic=[]
 for i in range(B):
  cp=[j for j in range(B) if ids[j]==ids[i] and j!=i];cn=[j for j in range(B) if ids[j]!=ids[i]]
  hp=[B+j for j,x in enumerate(mids) if x==ids[i]];hn=[B+j for j,x in enumerate(mids) if x!=ids[i]]
  cpj=cp[int(np.argmax(d[i,cp]))];cnj=cn[int(np.argmin(d[i,cn]))]
  basic.append(max(0,float(d[i,cpj])-float(d[i,cnj])+.3));pos=[cpj];neg=[cnj]
  if hp:
   j=hp[int(np.argmax(d[i,hp]))]
   if d[i,j]>d[i,cpj]:pos=[j]
   elif d[i,j]==d[i,cpj]:pos.append(j)
  if hn:
   j=hn[int(np.argmin(d[i,hn]))]
   if d[i,j]<d[i,cnj]:neg=[j]
   elif d[i,j]==d[i,cnj]:neg.append(j)
  margin=float(d[i,pos[0]])-float(d[i,neg[0]])+.3;v.append(max(0,margin))
  if margin>0:g[i,pos]=1/B/len(pos);g[i,neg]=-1/B/len(neg)
 return float(np.mean(v)),g,float(np.mean(basic))
widths={'baseline_only':3072,'fused':7680,'cnn':4608,'transformer':4608,'mamba':4608}
for fi,fold in enumerate(protocol['folds']):
 b0=baseline['folds'][fi];bind(b0['checkpoint'],b0['checkpoint_sha256']);payload=torch.load(b0['checkpoint'],map_location='cpu',weights_only=True);orig=payload['model_state_dict']
 assert payload['source_ids']==fold['source_ids'] and payload['heldout_ids']==fold['heldout_ids'] and payload['fold']==fi and not set(payload['source_ids'])&set(payload['heldout_ids'])
 assert state_sha(orig)==b0['training']['final_state_sha256']
 brefpath=Path(b0['checkpoint']).parent/'retrieval_arrays.pt';bind(brefpath,b0['retrieval']['retrieval_arrays_sha256']);bref=torch.load(brefpath,map_location='cpu',weights_only=True)
 out['source_checks'].append(dict(fold=fi,source_checkpoint_sha256=b0['checkpoint_sha256'],source_model_state_sha256=state_sha(orig),source_ids=len(fold['source_ids']),heldout_ids=len(fold['heldout_ids']),source_training_steps=len(b0['training']['steps']),source_training_scope_all_record_indices_valid=all(set(r['sampled_record_indices'])<=set(fold['source_record_indices']) for r in b0['training']['steps'])))
 for endpoint in ['control','smooth_ap']:
  ddir=RUN/'q1'/f'fold_{fi}_{endpoint}';receipt=load(ddir/'receipt.json');tr=load(ddir/'training.json');aud=[json.loads(l) for l in (ddir/'memory_steps.jsonl').read_text().splitlines()]
  assert receipt==summary['folds'][fi]['endpoints'][endpoint] and tr==receipt['training']
  for name in ['receipt.json','training.json','rankings.json']:bind(ddir/name)
  for name,proof in tr['audit_files'].items():bind(ddir/name,proof['sha256']);assert (ddir/name).stat().st_size==proof['bytes']
  bind(receipt['checkpoint'],receipt['checkpoint_sha256']);ck=torch.load(receipt['checkpoint'],map_location='cpu',weights_only=True)
  assert ck['binding']==receipt['initialization'] and ck['config_sha256']==summary['config_sha256'] and ck['fold']==fi and ck['source_ids']==fold['source_ids'] and ck['heldout_ids']==fold['heldout_ids']
  state={n:orig[a] for n,a in ck['baseline_aliases'].items()};assert all(n.startswith('baseline.') for n in state);assert not set(state)&set(ck['role_state_dict']);state.update(ck['role_state_dict'])
  assert state_sha(state)==tr['final_state_sha256']==receipt['strict_reload_state_sha256'];assert all(torch.isfinite(t).all() for t in state.values() if t.is_floating_point())
  trainable=set(receipt['initialization']['trainable_names']);frozen={n:t for n,t in state.items() if n not in trainable and not n.endswith(('running_mean','running_var','num_batches_tracked'))}
  # All baseline buffers belong to the frozen state even if their names end with BN suffixes.
  frozen.update({n:t for n,t in state.items() if n.startswith('baseline.')})
  frozen_sha=state_sha(frozen);assert frozen_sha==tr['frozen_state_after_sha256']
  assert state_sha(orig)==tr['signal_state_before_sha256']==tr['signal_state_after_sha256']
  r=dict(fold=fi,endpoint=endpoint,checkpoint_state_tensors=len(state),baseline_alias_tensors=len(ck['baseline_aliases']),role_checkpoint_tensors=len(ck['role_state_dict']),full_state_sha256=state_sha(state),frozen_state_sha256=frozen_sha,training_matrix_elements=0,anchors_checked=0,maximum_ap_error=0.,maximum_scalar_error=0.,maximum_smooth_derivative_error_float64=0.,maximum_smooth_derivative_error_float32=0.,maximum_hard_derivative_error_float64=0.,maximum_hard_derivative_error_float32=0.,derivative_distance_positions=0,self_distance_derivative_max=0.,retrieval={})
  with (ddir/'memory_distances.f32').open('rb') as f:
   for row,a in zip(tr['steps'],aud):
    n=a['distance_float_count'];assert f.tell()==a['distance_offset_bytes'];matrix=np.fromfile(f,dtype=np.float32,count=n).reshape(4,64,-1)
    assert np.isfinite(matrix).all() and np.all(matrix>=0);d=matrix[0];ids=a['identities'];mids=[x['identity'] for x in a['memory']]
    aps,grad=independent_ap_derivative(d,ids,mids);hard,hgrad,basic=independent_hard_derivative(d,ids,mids)
    ap_error=float(np.max(np.abs(aps-np.array(a['relation_objective']['per_anchor_smoothed_ap']))));r['maximum_ap_error']=max(r['maximum_ap_error'],ap_error);assert ap_error<2e-6
    scalar=max(abs(1-aps.mean()-a['relation_objective']['smooth_ap_loss']),abs(hard-a['relation_objective']['hard_loss']),abs(basic-a['original_triplet']));r['maximum_scalar_error']=max(r['maximum_scalar_error'],scalar);assert scalar<2e-6
    for dtype in [torch.float64,torch.float32]:
     current=torch.tensor(d[:,:64],dtype=dtype,requires_grad=True);history=torch.tensor(d[:,64:],dtype=dtype,requires_grad=True)
     hl,sl,ap=objective.paired_objectives(current,history,ids,mids)
     gg=torch.autograd.grad(sl,(current,history));actual=np.concatenate([v.numpy() for v in gg],1)
     hg=torch.autograd.grad(hl,(current,history),allow_unused=True);ah=np.concatenate([np.zeros_like(d[:,64:]) if v is None else v.numpy() for v in hg],1)
     k='float64' if dtype==torch.float64 else 'float32';error=float(np.max(np.abs(grad-actual)));he=float(np.max(np.abs(hgrad-ah)))
     r['maximum_smooth_derivative_error_'+k]=max(r['maximum_smooth_derivative_error_'+k],error);r['maximum_hard_derivative_error_'+k]=max(r['maximum_hard_derivative_error_'+k],he)
     assert error<(1e-12 if dtype==torch.float64 else 2e-6) and he==0
     r['self_distance_derivative_max']=max(r['self_distance_derivative_max'],float(np.max(np.abs(np.diag(actual[:,:64])))))
    r['training_matrix_elements']+=n;r['anchors_checked']+=64;r['derivative_distance_positions']+=d.size
   assert f.read()==b''
  arrpath=ddir/'retrieval_arrays.pt';bind(arrpath,receipt['retrieval']['retrieval_arrays_sha256']);arrays=torch.load(arrpath,map_location='cpu',weights_only=True);ranks=load(ddir/'rankings.json');positions=[q['gallery_position'] for q in fold['query_rows']]
  assert arrays['query_gallery_positions']==positions and arrays['gallery_record_indices']==fold['gallery_record_indices']
  assert torch.equal(arrays['features']['baseline_only'],bref['features']) and torch.equal(arrays['distances']['baseline_only'],bref['distances'])
  assert torch.equal(arrays['features']['fused'][:,:3072],arrays['features']['baseline_only'])
  torch.set_num_threads(56)
  for name,width in widths.items():
   values=arrays['features'][name];saved=arrays['distances'][name];assert values.shape==(len(fold['gallery_record_indices']),width) and torch.isfinite(values).all() and torch.isfinite(saved).all()
   unit=torch.nn.functional.normalize(values.float(),dim=1);q=unit[positions];dist=q.square().sum(1,keepdim=True)+unit.square().sum(1)[None];dist.addmm_(q,unit.T,beta=1,alpha=-2)
   assert torch.equal(dist,saved) and np.argsort(saved.numpy(),axis=1).tolist()==ranks[name]
   # Independent float64 vector algebra to characterize rounding, without replacing the fixed float32 ranking.
   v=values.double().numpy();u=v/np.sqrt(np.sum(v*v,axis=1,keepdims=True));df=np.sum((u[positions,None,:]-u[None,:,:])**2,axis=2)
   maxerr=float(np.max(np.abs(df-saved.numpy())));orders=np.argsort(df,axis=1);different=int(np.sum(orders!=np.array(ranks[name])))
   r['retrieval'][name]=dict(features_shape=list(values.shape),distance_shape=list(saved.shape),exact_fp32_replay=True,maximum_float64_distance_difference=maxerr,float64_different_rank_positions=different,rank_positions=saved.numel())
  torch.set_num_threads(1);out['endpoints'].append(r)
assert not torch.cuda.is_initialized()
out['elapsed_seconds']=time.perf_counter()-started;out['completed_at']=datetime.datetime.now().astimezone().isoformat();out['torch_version']=torch.__version__;out['numpy_version']=np.__version__
print(json.dumps(out,ensure_ascii=False,allow_nan=False))
