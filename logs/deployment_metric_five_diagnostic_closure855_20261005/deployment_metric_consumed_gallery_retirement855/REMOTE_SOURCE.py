from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/deployment_metric_consumed_gallery_retirement_20261005_855'
assert not journal.exists()
assert json.loads((root/'logs/deployment_metric_pending_fixed_best_launch_20261005_854/EXIT.json').read_text())['exit_code']==0
seal=json.loads((root/'refine-logs/deployment_metric_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(seal['artifact_sha256'])==271
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[]
for r in [{'diagnosis': 'results/global_task_role_fixed_best_20261004_v1/RGBNT201_semantic/DIAGNOSIS.json', 'diagnosis_sha256': '4be002804ea4cc710e41b04fa5ebee6c93381e780c64ed13e15618a265108f44', 'target': 'results/global_task_role_fixed_best_20261004_v1/RGBNT201_semantic/gallery_features.pt', 'bytes': 20565681, 'sha256': 'c48e00c0adb044574f8a52e6359874211abddd0cb22d8d1e8b54ca2456a53dbc'}, {'diagnosis': 'results/global_task_role_fixed_best_20261004_v1/RGBNT201_native/DIAGNOSIS.json', 'diagnosis_sha256': '9719c624752f1c1d538f445f83e5700f408d4df28b809928844a2d435ab280cf', 'target': 'results/global_task_role_fixed_best_20261004_v1/RGBNT201_native/gallery_features.pt', 'bytes': 20572924, 'sha256': 'cfb97d36a69bebc8c44d98c6a418d3794ecb682f74cf36be260857f79d2d1f9a'}, {'diagnosis': 'results/global_task_role_fixed_best_20261004_v1/MSVR310_semantic/DIAGNOSIS.json', 'diagnosis_sha256': '4c1798f8df5275b266a8869de70acd97f55bd7610eababacce2efb21149c4544', 'target': 'results/global_task_role_fixed_best_20261004_v1/MSVR310_semantic/gallery_features.pt', 'bytes': 25952305, 'sha256': 'b35a8f5e3e72aae3aefd80c65514ab5fa6125339e1044d8862fcd2adec38355d'}, {'diagnosis': 'results/global_task_role_fixed_best_20261004_v1/MSVR310_native/DIAGNOSIS.json', 'diagnosis_sha256': 'b7221cde7a6dbcc67ca81b742342d14033fd7b46373e750f94a47eae028b306a', 'target': 'results/global_task_role_fixed_best_20261004_v1/MSVR310_native/gallery_features.pt', 'bytes': 25961340, 'sha256': '04184dc41e41e61d2477f0317123cddb9cc16e1e9d7cb8adc7bcf55c87998664'}, {'diagnosis': 'results/global_task_role_fixed_best_20261004_v1/RGBNT100_semantic/DIAGNOSIS.json', 'diagnosis_sha256': '65d0fc24a5d0bb05a84b07b96ec3f7a5b50066638edc30a53324d6824f0518c0', 'target': 'results/global_task_role_fixed_best_20261004_v1/RGBNT100_semantic/gallery_features.pt', 'bytes': 210914225, 'sha256': 'e14a9d488a84e16b98d3087e49e87661e239543a82ef68358e9968e6687da9ca'}, {'diagnosis': 'results/global_task_role_fixed_best_20261004_v1/RGBNT100_native/DIAGNOSIS.json', 'diagnosis_sha256': '9e4697dd2f1d5785a40e8e13f77cc79c9fe21c9958474e42b731d8db0ba3d497', 'target': 'results/global_task_role_fixed_best_20261004_v1/RGBNT100_native/gallery_features.pt', 'bytes': 210983420, 'sha256': 'd85da71b4654c005d0ab8c23352713bd44dd592aaf11aca0f0caffd9eac333f9'}]:
 target=root/r['target'];diagnosis=root/r['diagnosis']
 assert target.resolve().is_relative_to((root/'results/global_task_role_fixed_best_20261004_v1').resolve())
 assert target.name=='gallery_features.pt' and str(target) not in seal['artifact_sha256']
 assert sha(diagnosis)==r['diagnosis_sha256'] and json.loads(diagnosis.read_text())['status']=='COMPLETE'
 exists=target.is_file()
 if exists:assert target.stat().st_size==r['bytes'] and sha(target)==r['sha256']
 rows.append(dict(r,present_before=exists))
journal.mkdir();(journal/'PREPARE.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),rows=rows),indent=2)+'\n')
before=shutil.disk_usage(root).free
for r in rows:
 if r['present_before']:(root/r['target']).unlink()
 assert not (root/r['target']).exists()
after=shutil.disk_usage(root).free
record=dict(status='CONSUMED_HISTORICAL_GALLERY_ONLY_RETIREMENT_COMPLETE',at=datetime.now().astimezone().isoformat(),rows=rows,
 retired_files=sum(r['present_before'] for r in rows),retired_bytes=sum(r['bytes'] for r in rows if r['present_before']),
 disk_free_before=before,disk_free_after=after,
 boundary='Only already diagnosed historical gallery_features caches, not in current271 dependencies. Every formal best, initializer, original evaluation distances, query features, DIAGNOSIS and source retained. No new model, current cache, failure weight, recursive deletion or scientific result change.')
(journal/'RETIREMENT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
