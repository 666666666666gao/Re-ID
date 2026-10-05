targets=['clean_clip_joint_20261002_v1_clean_clip_global_only_RGBNT201_seed42_full', 'clean_clip_joint_20261002_v1_clean_clip_roles_RGBNT201_seed42_full', 'shared_private_evidence_20261001_v2_shared_private_global_only_RGBNT201_seed42_full', 'shared_private_evidence_20261001_v2_shared_private_coupled_roles_RGBNT201_seed42_full', 'visual_update_control_20261001_v1_visual_update_frozen_global_only_RGBNT201_seed42_full', 'visual_update_control_20261001_v1_visual_update_low_lr_global_only_RGBNT201_seed42_full', 'visual_update_control_20261001_v1_visual_update_low_lr_roles_RGBNT201_seed42_full', 'native_detail_20261002_v1_clean_clip_high_MSVR310_seed42_full', 'visual_update_control_20261001_v1_visual_update_frozen_roles_MSVR310_seed42_full', 'visual_update_control_20261001_v1_visual_update_low_lr_roles_MSVR310_seed42_full']
from pathlib import Path
from datetime import datetime
import json,hashlib,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/dominated_closed_weight_retirement861_20261006'
assert not journal.exists()
assert not (root/'logs/row_mass_role_transport_v2_20261006_860').exists()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
sources=json.loads((root/'refine-logs/row_mass_role_transport_v2/SOURCE_SCOPE.json').read_text())['source_sha256']
protected=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
assert len(sources)==366 and len(protected)==187
assert all(sha(root/n)==d for n,d in sources.items()) and all(sha(Path(n))==d for n,d in protected.items())
winner_names={'RGBNT201':'region_reconstruction_v1_20261005_858_full_native_RGBNT201',
 'MSVR310':'metric_feature_scale_20261003_v1_full_metric_raw_MSVR310'}
def closed(name):
 folder=root/'trained-model'/name;assert folder.resolve().is_relative_to((root/'trained-model').resolve())
 t=json.loads((folder/'training.json').read_text());receipt=json.loads((folder/'official_metrics.json').read_text())
 assert [r['epoch'] for r in t['history']]==list(range(1,51)) and receipt['status']=='COMPLETE'
 best=next(row for row in t['history'] if row['epoch']==t['best_epoch'])
 assert best['official_fused']==receipt['metrics']
 assert sha(folder/'best_map.pth')==receipt['checkpoint_sha256']
 assert sha(folder/'official_distances.pt')==receipt['distance_sha256']
 return dict(directory=str(folder),best_epoch=t['best_epoch'],metrics=receipt['metrics'],
  artifact_sha256={str(folder/n):sha(folder/n) for n in ('training.json','official_metrics.json','official_distances.pt','best_map.pth')},
  weight_bytes=(folder/'best_map.pth').stat().st_size)
winners={dataset:closed(name) for dataset,name in winner_names.items()}
assert winners['RGBNT201']['artifact_sha256'][winners['RGBNT201']['directory']+'/best_map.pth']=='9b0b9da90e0ac5b231e73b95d712ed827d88cd0860f220bd2b412636e56062eb'
assert winners['MSVR310']['artifact_sha256'][winners['MSVR310']['directory']+'/best_map.pth']=='47a0a9928095c957b0331d46e5955ed5ca22eea678f915c1ca440461c999c683'
rows=[]
for name in targets:
 row=closed(name);path=Path(row['directory'])/'best_map.pth'
 assert str(path) not in protected and str(path.relative_to(root)) not in protected and str(path.relative_to(root)) not in sources
 dataset='RGBNT201' if 'RGBNT201' in name else 'MSVR310'
 winner=winners[dataset]
 assert all(winner['metrics'][k]>=v for k,v in row['metrics'].items())
 assert winner['metrics']['mAP']>row['metrics']['mAP']
 rows.append(dict(target=row,retained_winner=winner))
before=shutil.disk_usage(root).free
journal.mkdir();(journal/'PREPARE.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),rows=rows,free_before=before),indent=2)+'\n')
for row in rows:
 target=row['target'];path=Path(target['directory'])/'best_map.pth'
 assert sha(path)==target['artifact_sha256'][str(path)]
 path.unlink();assert not path.exists()
 with (journal/'deleted.jsonl').open('a') as stream:stream.write(json.dumps(dict(path=str(path),bytes=target['weight_bytes'],sha256=target['artifact_sha256'][str(path)]))+'\n')
 assert all(sha(n)==d for n,d in target['artifact_sha256'].items() if n!=str(path))
 assert all(sha(n)==d for n,d in row['retained_winner']['artifact_sha256'].items())
assert all(sha(root/n)==d for n,d in sources.items()) and all(sha(Path(n))==d for n,d in protected.items())
result=dict(status='TEN_DOMINATED_CLOSED_WEIGHTS_RETIRED',at=datetime.now().astimezone().isoformat(),
 removed_count=len(rows),removed_bytes=sum(r['target']['weight_bytes'] for r in rows),free_before=before,free_after=shutil.disk_usage(root).free,
 rows=rows,preserved_source_count=len(sources),preserved_control_artifacts=len(protected),
 boundary='Explicit ten closed self-trained201/MSVR unsupported/obsolete bests, allfourmetrics dominated. Actual50/receipt/weight/distance checked. No active or new-control dependency. Original text and distances retained; binary reload paths retired. No NN, 2025 or power-temperature action.')
(journal/'RETIREMENT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
