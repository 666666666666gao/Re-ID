from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
target=(root/'trained-model/visual_update_control_20261001_v1_visual_update_low_lr_roles_RGBNT100_seed42_full/best_map.pth').resolve()
winner=(root/'trained-model/shared_private_evidence_20261001_v2_shared_private_coupled_roles_RGBNT100_seed42_full/best_map.pth').resolve()
protected=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
assert str(target) not in protected and len(protected)==187
rows=[]
for p in (target,winner):
 assert p.is_relative_to((root/'trained-model').resolve()) and p.name=='best_map.pth'
 training=json.loads((p.parent/'training.json').read_text());receipt=json.loads((p.parent/'official_metrics.json').read_text())
 assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and receipt['status']=='COMPLETE'
 assert [h['epoch'] for h in training['history']]==list(range(1,51))
 assert receipt['dataset']=='RGBNT100' and receipt['seed']==42 and receipt['training_epochs']==50
 assert receipt['independent_upstream_metrics_equal'] and not receipt['reranking']
 assert sha(p)==receipt['checkpoint_sha256'] and sha(p.parent/'official_distances.pt')==receipt['distance_sha256']
 best=max(training['history'],key=lambda h:(h['official_fused']['mAP'],h['epoch']))
 assert best['epoch']==receipt['selected_epoch']==training['best_epoch']
 assert all(abs(best['official_fused'][k]-v)<1e-5 for k,v in receipt['metrics'].items())
 rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),training_sha256=sha(p.parent/'training.json'),receipt_sha256=sha(p.parent/'official_metrics.json'),metrics=receipt['metrics'],protocol_sha256=receipt['protocol_sha256'],baseline_sha256=receipt['baseline_sha256'],schema=training['schema']))
a,b=rows
assert a['protocol_sha256']==b['protocol_sha256'] and a['baseline_sha256']==b['baseline_sha256']
assert all(b['metrics'][k]>=a['metrics'][k] for k in ('mAP','Rank-1','Rank-5','Rank-10')) and b['metrics']['mAP']>a['metrics']['mAP']
current=root/'logs/signal_selection_reference_v1_20261006_868'
initializers=[json.loads(p.read_text()) for p in (current/'initialization').glob('*.json')];assert len(initializers)==9
assert all(str(target) not in json.dumps(r) for r in initializers)
assert target.stat().st_nlink==1
source=json.loads((root/'refine-logs/signal_selection_reference_v1/SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(source)==374 and all(sha(root/p)==s for p,s in source.items())
assert all(Path(p).exists() for p in protected)
out=root/'logs/selection_old_rgb100_role_retirement_20261006_868';assert not out.exists();out.mkdir()
before=shutil.disk_usage(root).free
qualification=dict(status='QUALIFIED_BEFORE_DELETE',at=datetime.now().astimezone().isoformat(),target=a,retained_winner=b,free_before_bytes=before,boundary='ClosedownoldRGB100roleweight all4metricinferior toretainednewerrolemodel withsame protocol/baseline/seed/full50. Differentarchitecture/flow, notcausalscienceclaim. Notcurrent9initializer/187RAW; alltexts/distances retained, originalbinarypath explicitlyretired. Authors andcurrentstudy untouched.')
(out/'QUALIFIED.json').write_text(json.dumps(qualification,indent=2)+'\n')
assert sha(target)==a['sha256']
(out/'RETIREMENT.json').write_text(json.dumps(dict(path=str(target),sha256=a['sha256'],bytes=a['bytes'],at=datetime.now().astimezone().isoformat(),retained=str(winner)),indent=2)+'\n')
target.unlink()
assert not target.exists() and sha(winner)==b['sha256']
assert sha(winner.parent/'training.json')==b['training_sha256'] and sha(winner.parent/'official_metrics.json')==b['receipt_sha256']
assert all(Path(p).exists() for p in protected) and all(sha(root/p)==s for p,s in source.items())
result=dict(status='COMPLETE',at=datetime.now().astimezone().isoformat(),retired=1,logical_retired_bytes=a['bytes'],free_before_bytes=before,free_after_bytes=shutil.disk_usage(root).free,target=a,retained_winner=b,protected_artifacts=187,source_count=374,out_dir=str(out))
(out/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(result=result,files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()})))
