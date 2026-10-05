from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/obsolete_scale_weight_retirement_20261005_856'
assert not journal.exists()
dependencies=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
sources=json.loads((root/'refine-logs/semantic_capacity_control_v1/SOURCE_SCOPE.json').read_text())['source_sha256']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[{'directory': '/data/gaob/Re-ID/Trifusion/trained-model/metric_feature_scale_20261003_v1_full_metric_raw_RGBNT201', 'dataset': 'RGBNT201', 'condition': {'recipe': 'metric_raw', 'seed': 42, 'epochs': 50, 'public_clip_sha256': '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f', 'initialization_sha256': 'b953dd0373c05cea1041ab2cc8315845d6aedc56f125bf2058df72c53e9a8fb1'}, 'training_status': 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE', 'evaluation_status': 'COMPLETE', 'epochs': 50, 'best_epoch': 38, 'metrics': {'mAP': 59.73434400981085, 'Rank-1': 60.28708219528198, 'Rank-5': 72.96651005744934, 'Rank-10': 82.41626620292664}, 'best_path': '/data/gaob/Re-ID/Trifusion/trained-model/metric_feature_scale_20261003_v1_full_metric_raw_RGBNT201/best_map.pth', 'best_bytes': 346781649, 'best_sha256': '9aa7151f67ac353a32e5ff3c0547d7cf024a8be64db99a807065832b0cb0b945', 'checkpoint_receipt_sha256': '9aa7151f67ac353a32e5ff3c0547d7cf024a8be64db99a807065832b0cb0b945', 'training_sha256': '5ed89423a68b9d5825cf1cf31d7884d6940e5d0a8c84fea7363bd940c8b6199e', 'evaluation_sha256': '8f8fe7110b1f427eebac9eb750a6fb44f2580583be6a40a6c8de27189655ec73', 'protected_current_raw187': False, 'original_distance_present': True}, {'directory': '/data/gaob/Re-ID/Trifusion/trained-model/metric_feature_scale_20261003_v1_full_metric_raw_RGBNT100', 'dataset': 'RGBNT100', 'condition': {'recipe': 'metric_raw', 'seed': 42, 'epochs': 50, 'public_clip_sha256': '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f', 'initialization_sha256': '0fa6fd47b514bbb4c54ee6254fa3b92ca6621e4b74e74eefea845d74d7c352f6'}, 'training_status': 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE', 'evaluation_status': 'COMPLETE', 'epochs': 50, 'best_epoch': 48, 'metrics': {'mAP': 75.45198460760588, 'Rank-1': 94.75218653678894, 'Rank-5': 95.5102026462555, 'Rank-10': 95.86005806922913}, 'best_path': '/data/gaob/Re-ID/Trifusion/trained-model/metric_feature_scale_20261003_v1_full_metric_raw_RGBNT100/best_map.pth', 'best_bytes': 345310725, 'best_sha256': 'd2a3865a516eff38ee30078d30245831c9914d3d271a35a8a0934dc243960651', 'checkpoint_receipt_sha256': 'd2a3865a516eff38ee30078d30245831c9914d3d271a35a8a0934dc243960651', 'training_sha256': '3a9269fe4238bd7537de6662bca4c9e549385f7650e09f1fcdffabcc30311d6e', 'evaluation_sha256': '968af5c200c5dc5b5ee363bceb7fe0bf6645a612cc5b9ce7b0cb959fb746190a', 'protected_current_raw187': False, 'original_distance_present': True}, {'directory': '/data/gaob/Re-ID/Trifusion/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT100', 'dataset': 'RGBNT100', 'condition': {'recipe': 'raw', 'seed': 42, 'epochs': 50, 'public_clip_sha256': '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f', 'initialization_sha256': 'e824ca9a8c3bed82d00bacf03d681b51b9e7391320c952fb969d33667325f2c8'}, 'training_status': 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE', 'evaluation_status': 'COMPLETE', 'epochs': 50, 'best_epoch': 47, 'metrics': {'mAP': 75.19293649431079, 'Rank-1': 94.40233111381531, 'Rank-5': 95.91836929321289, 'Rank-10': 96.38484120368958}, 'best_path': '/data/gaob/Re-ID/Trifusion/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT100/best_map.pth', 'best_bytes': 345310725, 'best_sha256': '2f8a30483fd598ee1639cefa7dcb6c0b1fff50a3d56786b68b2d6db25f7396a2', 'checkpoint_receipt_sha256': '2f8a30483fd598ee1639cefa7dcb6c0b1fff50a3d56786b68b2d6db25f7396a2', 'training_sha256': '2f6507087db4f7aa80f1b54175036978b32ca1219c6693d291fa6e4a6deb9520', 'evaluation_sha256': '08e42d31c2fd322c04a1e4ce0e095618969a6c855742ea3a3bec35489a71adbb', 'protected_current_raw187': False, 'original_distance_present': True}, {'directory': '/data/gaob/Re-ID/Trifusion/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT201', 'dataset': 'RGBNT201', 'condition': {'recipe': 'raw', 'seed': 42, 'epochs': 50, 'public_clip_sha256': '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f', 'initialization_sha256': '76a7f92b5701a149ba0cd0dda0d8c295f233d4e9a5c036de7157ecc5efef6eba'}, 'training_status': 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE', 'evaluation_status': 'COMPLETE', 'epochs': 50, 'best_epoch': 26, 'metrics': {'mAP': 59.68789908355065, 'Rank-1': 60.88516712188721, 'Rank-5': 73.20573925971985, 'Rank-10': 80.62201142311096}, 'best_path': '/data/gaob/Re-ID/Trifusion/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT201/best_map.pth', 'best_bytes': 346781649, 'best_sha256': '98d93c8520d5419d669871bacc8800ecd226bb576e6b1d6e81b04e434154df49', 'checkpoint_receipt_sha256': '98d93c8520d5419d669871bacc8800ecd226bb576e6b1d6e81b04e434154df49', 'training_sha256': 'ffa12df9943227789cbe14fb31911335955076ce02b7e297f9a2bdbedc719750', 'evaluation_sha256': '59c871bcbf36a3e688df9662d58fadf1455511e4f884acd92c5e102cc81312ce', 'protected_current_raw187': False, 'original_distance_present': True}]
for row in rows:
 target=Path(row['best_path']);folder=target.parent
 assert target.resolve().is_relative_to((root/'trained-model').resolve()) and target.name=='best_map.pth'
 assert folder.name.startswith(('training_feature_scale_20261003_v2_full_raw_','metric_feature_scale_20261003_v1_full_metric_raw_'))
 assert str(target) not in dependencies and target.relative_to(root).as_posix() not in sources
 assert sha(target)==row['best_sha256'] and target.stat().st_size==row['best_bytes']
 assert sha(folder/'training.json')==row['training_sha256'] and sha(folder/'official_metrics.json')==row['evaluation_sha256']
 t=json.loads((folder/'training.json').read_text());e=json.loads((folder/'official_metrics.json').read_text())
 assert e['dataset'] in ('RGBNT201','RGBNT100') and e['condition']['recipe'] in ('raw','metric_raw')
 assert len(t['history'])==50 and e['status']=='COMPLETE' and t['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
 assert t['best_epoch']==e['selected_epoch'] and e['checkpoint_sha256']==row['best_sha256']
 assert sha(folder/'official_distances.pt')==e['distance_sha256']
 assert sha(folder/'best_epoch_distances.pt')==e['training_best_distance_sha256']
 row['retained_formal_distance_sha256']=e['distance_sha256']
 row['retained_best_epoch_distance_sha256']=e['training_best_distance_sha256']
 assert not Path('/proc/'+str(t.get('controller_pid',-1))).exists()
journal.mkdir();(journal/'PREPARE.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),rows=rows),indent=2)+'\n')
before=shutil.disk_usage(root).free
for row in rows:
 target=Path(row['best_path']);target.unlink();assert not target.exists()
 assert (target.parent/'official_distances.pt').is_file() and (target.parent/'official_metrics.json').is_file()
after=shutil.disk_usage(root).free
value=dict(status='FOUR_OBSOLETE_NEGATIVE_SCALE_BESTS_RETIRED',at=datetime.now().astimezone().isoformat(),
 rows=rows,retired_files=4,retired_bytes=sum(r['best_bytes'] for r in rows),disk_free_before=before,disk_free_after=after,
 boundary='Four finished F2/F3 raw or metric_raw candidates on201/100 rejected by primary mAP. Not required by current345/187. Original metrics/history/distances/SHA preserved; old weight-reload paths explicitly retired. No normalized controls, MSVR positive candidates, author weights, initialization or current best removed. Current NN/source/GPU untouched.')
(journal/'RETIREMENT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
