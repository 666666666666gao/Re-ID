from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
trained=(root/'trained-model').resolve()
journal=root/'logs/rejected_candidate_best_retirement_20261005_858'
assert not journal.exists()
controls=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
protected=controls['artifact_sha256']
sources=json.loads((root/'refine-logs/semantic_capacity_control_v1/SOURCE_SCOPE.json').read_text())['source_sha256']
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==d for n,d in protected.items())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in sources.items())
targets=[]
for dataset in ('RGBNT201','RGBNT100'):
    kept=f'clean_clip_joint_20261002_v1_clean_clip_roles_{dataset}_seed42_full'
    targets.extend((f'native_detail_20261002_v1_clean_clip_{resolution}_{dataset}_seed42_full',kept)
                   for resolution in ('high','low'))
for dataset in ('RGBNT201','MSVR310','RGBNT100'):
    targets.append((f'shared_private_evidence_20261001_v2_shared_private_separated_roles_{dataset}_seed42_full',
                    f'shared_private_evidence_20261001_v2_shared_private_coupled_roles_{dataset}_seed42_full'))
for dataset in ('RGBNT201','RGBNT100'):
    targets.append((f'visual_update_control_20261001_v1_visual_update_frozen_roles_{dataset}_seed42_full',
                    f'visual_update_control_20261001_v1_visual_update_low_lr_roles_{dataset}_seed42_full'))
assert len(targets)==9
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
rows=[]
for name,retained in targets:
    folder=trained/name;weight=(folder/'best_map.pth').resolve()
    assert weight.is_relative_to(trained) and weight.parent==folder and weight.name=='best_map.pth'
    assert str(weight) not in protected and str(weight.relative_to(root)) not in sources
    receipt=folder/'official_metrics.json';history=folder/'training.json'
    metrics=json.loads(receipt.read_text());training=json.loads(history.read_text())
    assert metrics['status']=='COMPLETE' and training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert [r['epoch'] for r in training['history']]==list(range(1,51))
    assert digest(weight)==metrics['checkpoint_sha256']
    distances=folder/'official_distances.pt';best_distances=folder/'best_epoch_distances.pt'
    assert digest(distances)==metrics['distance_sha256']
    assert digest(best_distances)==metrics['training_best_distance_sha256']
    kept=trained/retained
    kept_receipt=json.loads((kept/'official_metrics.json').read_text())
    assert kept_receipt['metrics']['mAP']>metrics['metrics']['mAP']
    assert digest(kept/'best_map.pth')==kept_receipt['checkpoint_sha256']
    preserved={str(p):digest(p) for p in (receipt,history,distances,best_distances)}
    rows.append(dict(path=str(weight),bytes=weight.stat().st_size,sha256=metrics['checkpoint_sha256'],
        mAP=metrics['metrics']['mAP'],retained_better_weight=str(kept/'best_map.pth'),
        retained_mAP=kept_receipt['metrics']['mAP'],preserved_artifacts=preserved))
journal.mkdir()
before=shutil.disk_usage(root).free
prepare=dict(status='VERIFIED_LOWER_MAP_CANDIDATES',prepared_at=datetime.now().astimezone().isoformat(),
    rows=rows,free_bytes_before=before,control_count=len(protected),source_count=len(sources))
(journal/'PREPARE.json').write_text(json.dumps(prepare,indent=2)+'\n')
for row in rows:Path(row['path']).unlink()
assert all(not Path(row['path']).exists() for row in rows)
assert all(digest(Path(n))==d for row in rows for n,d in row['preserved_artifacts'].items())
assert all(digest(Path(n))==d for n,d in protected.items())
assert all(digest(root/n)==d for n,d in sources.items())
result=dict(prepare,status='RETIRED_COMPLETED_LOWER_MAP_BESTS',retired_at=datetime.now().astimezone().isoformat(),
    bytes_retired=sum(r['bytes'] for r in rows),free_bytes_after=shutil.disk_usage(root).free,
    boundary='Only nine named completed, unselected self-trained candidates. Better controls and current187/345 retained. Original metric/distance/history evidence unchanged; these nine binary reload paths are now retired. No other project or power/temperature operation.')
(journal/'RETIREMENT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
