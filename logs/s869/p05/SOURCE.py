from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,torch
root=Path('/data/gaob/Re-ID/Trifusion')
assert not torch.cuda.is_available()
protected=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def collect(name):
 p=(root/'trained-model'/name/'best_map.pth').resolve()
 assert p.is_relative_to((root/'trained-model').resolve()) and str(p) not in protected
 t=p.parent/'training.json';m=p.parent/'official_metrics.json'
 training=json.loads(t.read_text());receipt=json.loads(m.read_text())
 assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and receipt['status']=='COMPLETE'
 assert [h['epoch'] for h in training['history']]==list(range(1,51))
 digest=sha(p)
 assert digest==receipt['checkpoint_sha256']
 assert sha(p.parent/'official_distances.pt')==receipt['distance_sha256']
 assert 'training_best_distance_sha256' not in receipt
 assert receipt['independent_upstream_metrics_equal'] and not receipt['reranking']
 assert receipt['training_epochs']==50
 payload=torch.load(p,map_location='cpu',weights_only=True)
 assert payload['epoch']==training['best_epoch']==receipt['selected_epoch']
 assert payload['metrics']==next(h['official_fused'] for h in training['history'] if h['epoch']==training['best_epoch'])
 assert all(abs(payload['metrics'][k]-v)<1e-5 for k,v in receipt['metrics'].items())
 facts=dict(path=str(p),bytes=p.stat().st_size,sha256=digest,training_sha256=sha(t),receipt_sha256=sha(m),dataset=training['dataset'],metrics=receipt['metrics'],protocol_sha256=payload['protocol_sha256'],baseline_sha256=payload['baseline_sha256'],checkpoint_schema=payload['schema'],condition=payload['condition'],state_keys=len(payload['state']))
 return facts,payload
pairs=[]
for dataset in ('MSVR310','RGBNT100'):
 target,payload=collect(f'shared_private_evidence_20261001_v2_shared_private_global_only_{dataset}_seed42_full')
 winner,retained=collect(f'visual_update_control_20261001_v1_visual_update_low_lr_global_only_{dataset}_seed42_full')
 assert target['dataset']==winner['dataset'] and target['protocol_sha256']==winner['protocol_sha256'] and target['baseline_sha256']==winner['baseline_sha256']
 assert target['metrics']==winner['metrics']
 assert payload['state'].keys()==retained['state'].keys()
 assert all(a.dtype==retained['state'][k].dtype and a.shape==retained['state'][k].shape and torch.equal(a,retained['state'][k]) for k,a in payload['state'].items())
 pairs.append(dict(reason='ALL_STATE_TENSORS_AND_BUFFERS_EXACT_EQUAL_DUPLICATE_CLOSED_MODEL',target=target,retained=winner,strict_original_binary_replay_retired=True,metadata_difference='Separate study schema/condition remains in originaltext; retainedmodelstate is identical, no filealias/fallback ormetadatarewrite.'))
 del payload,retained
target,payload=collect('visual_update_control_20261001_v1_visual_update_frozen_global_only_RGBNT100_seed42_full')
winner,retained=collect('visual_update_control_20261001_v1_visual_update_low_lr_global_only_RGBNT100_seed42_full')
assert target['dataset']==winner['dataset'] and target['protocol_sha256']==winner['protocol_sha256'] and target['baseline_sha256']==winner['baseline_sha256']
assert all(winner['metrics'][k]>=target['metrics'][k] for k in ('mAP','Rank-1','Rank-5','Rank-10')) and winner['metrics']['mAP']>target['metrics']['mAP']
pairs.append(dict(reason='CLOSED_OLD_SAME_ARCHITECTURE_GLOBAL_ONLY_CONTROL_ALL4_METRICS_INFERIOR',target=target,retained=winner,strict_original_binary_replay_retired=True,metadata_difference='Frozen and lowLR controls remain scientificdistinct; no relabeling. Text/curves/officialdistances retained; only obsolete inferior ownweight eligible.'))
current=root/'logs/signal_selection_reference_v1_20261006_868'
initializers=[json.loads(p.read_text()) for p in (current/'initialization').glob('*.json')]
assert len(initializers)==9
for pair in pairs:
 assert all(pair['target']['path'] not in json.dumps(r) for r in initializers)
sources=json.loads((root/'refine-logs/signal_selection_reference_v1/SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(sources)==374 and all(sha(root/n)==s for n,s in sources.items())
assert len(protected)==187 and all(sha(p)==s for p,s in protected.items())
out=root/'logs/selection_redundant_global_qualification_20261006_868_r2'
assert not out.exists();out.mkdir()
result=dict(status='QUALIFIED_NOT_DELETED',at=datetime.now().astimezone().isoformat(),pairs=pairs,prospective_retired_bytes=sum(r['target']['bytes'] for r in pairs),free_bytes=shutil.disk_usage(root).free,protected_sha256=protected,source_sha256=sources,boundary='CPUstate comparison/oldclosedfirststrictandphysicalhash only. No new NNforward/update/evaluation/report/modelrewrite. Retainall authors/RAW187/currentninebest+probes and retainedoldglobalwinners. Currentinitializers do not reference targets. Originalweights mustbe markedretired, notsilentlypresumedreplayable.')
(out/'QUALIFIED.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(result=result,remote_qualification=str(out/'QUALIFIED.json'),qualification_sha256=sha(out/'QUALIFIED.json'))))
