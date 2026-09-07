from pathlib import Path
from collections import Counter
from datetime import datetime
import json,hashlib
import numpy as np
root=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
protocol_path=root/'protocols/msvr310_train_oof_v1.json'
protocol=json.loads(protocol_path.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
folds=[]
for fold in protocol['folds']:
 number=fold['fold']
 training_path=root/f'evidence/msvr310_trifusion_v1_complete/fold_{number}_training.json'
 training=json.loads(training_path.read_bytes())
 assert training['optimizer_steps']==260 and training['epochs']==20
 source=set(fold['source_record_indices'])
 exposure=Counter();counts=Counter();batches=[]
 for k,row in enumerate(training['steps']):
  ii=row['sampled_record_indices'];assert len(ii)==64 and set(ii)<=source
  rr=[protocol['records'][i] for i in ii]
  ids=np.array([r['identity'] for r in rr]);cams=np.array([r['camera'] for r in rr]);scenes=np.array([r['scene'] for r in rr])
  assert sorted(Counter(ids).values())==[8]*8
  allowed=cams[:,None]!=cams[None,:]
  assert allowed.any(axis=1).all()
  rng=np.random.default_rng(np.random.SeedSequence([42,number,k]))
  active=bool(rng.random()<.5);choices=rng.random(allowed.shape);choices[~allowed]=np.inf
  donors=choices.argmin(axis=1);coeff=rng.beta(.1,.1,size=64)
  positive=(ids[:,None]==ids[None,:]) & ~np.eye(64,dtype=bool)
  counts['batches']+=1;counts['active_batches']+=active
  counts['cross_camera_positive_exposures']+=int((positive & allowed).sum())
  counts['cross_scene_positive_exposures']+=int((positive & (scenes[:,None]!=scenes[None,:])).sum())
  counts['positive_exposures']+=int(positive.sum())
  counts['duplicate_record_positive_exposures']+=sum(n*(n-1) for n in Counter(ii).values())
  counts['donor_cross_camera_exposures']+=64
  counts['donor_cross_scene_exposures']+=int((scenes[donors]!=scenes).sum())
  counts['donor_same_identity_exposures']+=int((ids[donors]==ids).sum())
  exposure.update(ii)
  batches.append(dict(step=k,epoch=row['epoch'],record_indices=ii,style_plan=dict(fold=number,step=k,active=active,forced_active=False,donors=donors.tolist(),coefficients=coeff.tolist(),all_donors_cross_camera=True),camera_values=sorted(set(cams.tolist()))))
 assert set(exposure)==source
 folds.append(dict(fold=number,source_ids=len(fold['source_ids']),source_records=len(source),heldout_ids=len(fold['heldout_ids']),gallery=len(fold['gallery_record_indices']),query=len(fold['query_rows']),counts=dict(counts),minimum_cameras_per_batch=min(len(r['camera_values']) for r in batches),source_exposures={str(k):v for k,v in sorted(exposure.items())},training_sha256=sha(training_path),batches=batches))
out=dict(status='PASS_EXISTING_MSVR310_SOURCE_EXPOSURE_AND_CAMERA_DONOR_SUPPORT',checked_at=datetime.now().astimezone().isoformat(),scope='Read existing source training metadata only; not new sampler/image/gradient replay or training registration',protocol_sha256=sha(protocol_path),folds=folds,total_batches=sum(f['counts']['batches'] for f in folds),image_reads=0,model_forwards=0,optimizer_updates=0)
p=Path('C:/Users/gb/.codex_tmp/msvr310_style_source_readiness_20260907.json')
p.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({**{k:v for k,v in out.items() if k!='folds'},'folds':[{k:v for k,v in f.items() if k not in ('batches','source_exposures')} for f in folds],'file_bytes':p.stat().st_size}))

