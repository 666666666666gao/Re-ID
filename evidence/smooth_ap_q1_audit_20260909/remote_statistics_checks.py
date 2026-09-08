"""Independent full Q1 metadata/statistics and positive-distance-derivative census."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,time,datetime
import numpy as np
RUN=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
started=time.perf_counter();result={'status':'PASS','scope':'All 1560 Q1 fused distance matrices; saved role matrices checked for finiteness. Float64 analytic derivative census, not model or parameter gradients.','endpoints':[],'model_forwards':0,'optimizer_updates':0,'image_reads':0,'files':{}}
for fi in range(3):
 for e in ['control','smooth_ap']:
  p=RUN/'q1'/f'fold_{fi}_{e}';auds=[json.loads(l) for l in (p/'memory_steps.jsonl').read_text().splitlines()];tr=json.loads((p/'training.json').read_bytes());assert sha(p/'memory_distances.f32')==tr['audit_files']['memory_distances.f32']['sha256']
  result['files'][str(p/'memory_distances.f32')]=sha(p/'memory_distances.f32');counts=defaultdict(lambda:defaultdict(lambda:dict(count=0,negative=0,zero=0,positive=0)))
  print(json.dumps({'event':'census_endpoint','fold':fi,'endpoint':e}),flush=True)
  with (p/'memory_distances.f32').open('rb') as f:
   for a in auds:
    mat=np.fromfile(f,dtype=np.float32,count=a['distance_float_count']).reshape(4,64,-1);d=mat[0];ids=np.array(a['identities']);sc=np.array(a['scenes']);mids=np.array([m['identity'] for m in a['memory']]);msc=np.array([m['scene'] for m in a['memory']]);mp=ids[:,None]==mids[None,:];pos=(ids[:,None]==ids[None,:])&~np.eye(64,dtype=bool);neg=ids[:,None]!=ids[None,:];dc=d[:,:64];dm=d[:,64:]
    hp=np.max(np.where(pos,dc,-np.inf),axis=1);hn=np.min(np.where(neg,dc,np.inf),axis=1)
    if len(mids):
     mph=np.max(np.where(mp,dm,-np.inf),axis=1);mnh=np.min(np.where(~mp,dm,np.inf),axis=1);uhp=np.maximum(hp,mph);uhn=np.minimum(hn,mnh);harderp=int((mph>hp).sum());hardern=int((mnh<hn).sum())
    else:uhp=hp;uhn=hn;harderp=hardern=0
    computed=dict(memory_records=len(mids),memory_positive_pairs=int(mp.sum()),memory_negative_pairs=int((~mp).sum()),memory_cross_scene_positive_pairs=int((mp&(sc[:,None]!=msc[None,:])).sum()),memory_negative_violations_against_batch_hard_positive=int(((~mp)&(dm<hp[:,None]+np.float32(.3))).sum()),harder_positive_anchors=harderp,harder_negative_anchors=hardern,current_wrong_order_anchors=int((hp>=hn).sum()),expanded_wrong_order_anchors=int((uhp>=uhn).sum()),expanded_hinge_positive_anchors=int((uhp-uhn+np.float32(.3)>0).sum()),maximum_memory_age=max((m['age'] for m in a['memory']),default=0))
    for k,v in computed.items():assert v==a['statistics'][k],(p.name,a['step'],k,v,a['statistics'][k])
    allids=np.concatenate([ids,mids]);allsc=np.concatenate([sc,msc]);columns=np.arange(d.shape[1]);score=1-d.astype(np.float64)**2/2
    phases=['all','post_warmup' if a['replacement_active'] else 'warmup']+(['last65'] if a['step']>=196 else [])
    for i in range(64):
     positive=(allids==ids[i]);positive[i]=False;negative=allids!=ids[i];pp=np.flatnonzero(positive);cross=allsc!=sc[i]
     s=score[i];t=1/(1+np.exp(-(s[None,:]-s[pp,None])/.01));valid=(columns[None,:]!=i)&(columns[None,:]!=pp[:,None]);t[~valid]=0;rp=1+t[:,positive].sum(1);ra=1+t.sum(1);jac=-(positive[None,:]*ra[:,None]-rp[:,None])/(ra[:,None]**2*len(pp)*64);term=jac*t*(1-t)/.01;term[~valid]=0;gs=term.sum(0);gs[pp]-=term.sum(1);gd=-gs*d[i]
     inv=positive&(d[i]>np.min(d[i,negative]));nonmax=positive&(d[i]<np.max(d[i,positive]));masks=dict(all_positive=positive,cross_positive=positive&cross,inverted_positive=inv,inverted_nonmax=inv&nonmax,inverted_nonmax_cross=inv&nonmax&cross)
     for phase in phases:
      for name,mask in masks.items():
       v=gd[mask];r=counts[phase][name];r['count']+=len(v);r['negative']+=int((v<0).sum());r['zero']+=int((v==0).sum());r['positive']+=int((v>0).sum())
   assert f.read()==b''
  result['endpoints'].append(dict(endpoint=p.name,steps=len(auds),phases={phase:dict(v) for phase,v in counts.items()}))
result['elapsed_seconds']=time.perf_counter()-started;result['completed_at']=datetime.datetime.now().astimezone().isoformat();print(json.dumps(result))
