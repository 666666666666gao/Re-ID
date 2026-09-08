import collections,hashlib,json,math,random,sys
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/autodl-tmp/trifusion-v2/artifacts/msvr310_history_gradient_v1_seed42_a1b4777')
hashes={}
def get(path):
 p=Path(path);b=p.read_bytes();hashes[str(p)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return json.loads(b)
configs=['TriFusion-history-gradient-paired-v1.json','TriFusion-fresh-coordinate-paired-v1.json','TriFusion-instance-memory-paired-v1.json','TriFusion-source-style-paired-v1-r2.json','Signal-source-oof-v1.json']
checked=[]
for n in configs:
 config=get(ROOT/'configs/MSVR310'/n)
 for field in ('project_file_sha256','project_source_file_sha256','fixed_file_sha256'):
  for name,wanted in config.get(field,{}).items():
   p=Path(name) if Path(name).is_absolute() else ROOT/name;b=p.read_bytes();actual=hashlib.sha256(b).hexdigest();assert wanted==actual,(n,name)
   hashes[str(p)]={'bytes':len(b),'sha256':actual};checked.append(dict(config=n,field=field,path=str(p),sha256=actual))
 if 'signal_source_file_sha256' in config:
  for name,wanted in config['signal_source_file_sha256'].items():
   p=Path(config['signal_source'])/name;b=p.read_bytes();actual=hashlib.sha256(b).hexdigest();assert actual==wanted,name
   hashes[str(p)]={'bytes':len(b),'sha256':actual};checked.append(dict(config=n,field='signal_source_file_sha256',path=str(p),sha256=actual))
protocol=get(ROOT/'protocols/msvr310_train_oof_v1.json');meta=get(ROOT/'evidence/msvr310_style_t0_runtime_binding_20260907/msvr_style_metadata_remote_numpy_20260907.json');t0=get(RUN/'t0.json');m0=get(RUN/'m0/summary.json')
folds=[]
for f,fold in enumerate(protocol['folds']):
 rng=random.Random(42);nprng=np.random.RandomState(42);byid={}
 for i in fold['source_record_indices']:byid.setdefault(protocol['records'][i]['identity'],[]).append(i)
 all_batches=[]
 for epoch in range(20):
  queues={}
  for identity,indices in byid.items():
   vals=list(indices) if len(indices)>=8 else nprng.choice(indices,size=8,replace=True)
   rng.shuffle(vals);queues[identity]=[list(map(int,vals[k:k+8])) for k in range(0,len(vals)-7,8)]
  available=list(byid)
  while len(available)>=8:
   chosen=rng.sample(available,8);batch=[]
   for identity in chosen:
    batch+=queues[identity].pop(0)
    if not queues[identity]:available.remove(identity)
   all_batches.append(batch)
 assert len(all_batches)==260
 assert all_batches==[r['record_indices'] for r in meta['folds'][f]['batches']]
 queue=collections.OrderedDict();counts=[]
 for step,indices in enumerate(all_batches):
  for i,s in list(queue.items()):
   if step-s>8:del queue[i]
  selected=[step-s for i,s in queue.items() if i not in indices]
  assert t0['folds'][f]['steps'][step]==dict(step=step+1,historical_records=len(selected),ages=selected)
  counts.append(len(selected))
  if step>=65:
   for i in indices:queue.pop(i,None);queue[i]=step
   while len(queue)>512:queue.popitem(last=False)
 folds.append(dict(fold=f,source_sampler_batches_recreated=260,t0_queue_steps_recreated=260,historical_candidate_record_exposures=sum(counts),maximum_candidates=max(counts)))
overfit=[]
for end,item in m0['overfit'].items():
 tr=item['training'];k=len(protocol['folds'][0]['source_ids']);smooth=.1
 p=1-smooth+smooth/k;q=smooth/k;entropy=-p*math.log(p)-(k-1)*q*math.log(q)
 floor=entropy*(.25+3/12+3/12)
 ratio=(tr['steps'][-1]['loss']-floor)/(tr['steps'][0]['loss']-floor)
 assert abs(floor-item['gate']['minimum_loss'])<1e-12 and abs(ratio-item['gate']['loss_ratio'])<1e-12 and ratio<=.1
 overfit.append(dict(endpoint=end,minimum_loss=floor,first_loss=tr['steps'][0]['loss'],last_loss=tr['steps'][-1]['loss'],excess_loss_ratio=ratio))
print(json.dumps(dict(status='PASS_INDEPENDENT_CONTRACT_SAMPLER_T0_M0_CHECKS',hash_bound_checks=len(checked),bindings=checked,folds=folds,overfit=overfit,audited_input_hashes=hashes,model_forwards=0,optimizer_updates=0,official_image_reads=0),indent=2))
