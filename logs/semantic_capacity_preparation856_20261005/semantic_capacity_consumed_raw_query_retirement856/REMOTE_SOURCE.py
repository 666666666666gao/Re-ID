from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/semantic_capacity_consumed_raw_query_retirement_20261005_856'
assert not journal.exists()
assert json.loads((root/'logs/deployment_metric_pending_fixed_best_launch_20261005_854/EXIT.json').read_text())['exit_code']==0
seal=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(seal['artifact_sha256'])==187
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[]
for r in [{'diagnosis': 'results/global_task_role_fixed_best_20261004_v1/RGBNT201_semantic/DIAGNOSIS.json', 'diagnosis_sha256': '4be002804ea4cc710e41b04fa5ebee6c93381e780c64ed13e15618a265108f44', 'target': 'results/global_task_role_fixed_best_20261004_v1/RGBNT201_semantic/query_features.pt', 'bytes': 20565655, 'sha256': '31145c99c851df3afa80246391af80979d0750757763c61289386e89f60185f2'}, {'diagnosis': 'results/global_task_role_fixed_best_20261004_v1/RGBNT201_native/DIAGNOSIS.json', 'diagnosis_sha256': '9719c624752f1c1d538f445f83e5700f408d4df28b809928844a2d435ab280cf', 'target': 'results/global_task_role_fixed_best_20261004_v1/RGBNT201_native/query_features.pt', 'bytes': 20572894, 'sha256': '7a8e5aefe150abad471d44939897c9264c92de307c0d5dab5bbe752d17b25d00'}, {'diagnosis': 'results/global_task_role_fixed_best_20261004_v1/MSVR310_semantic/DIAGNOSIS.json', 'diagnosis_sha256': '4c1798f8df5275b266a8869de70acd97f55bd7610eababacce2efb21149c4544', 'target': 'results/global_task_role_fixed_best_20261004_v1/MSVR310_semantic/query_features.pt', 'bytes': 14539735, 'sha256': '9c01b2a26970de094f779fefdd6d9e5b2781546425ea56cd9e2093a3145c795e'}, {'diagnosis': 'results/global_task_role_fixed_best_20261004_v1/MSVR310_native/DIAGNOSIS.json', 'diagnosis_sha256': 'b7221cde7a6dbcc67ca81b742342d14033fd7b46373e750f94a47eae028b306a', 'target': 'results/global_task_role_fixed_best_20261004_v1/MSVR310_native/query_features.pt', 'bytes': 14545054, 'sha256': '76cc59e6a4fdb109972569b22e142b37f72706c370fcf4f21e391e60453b54c5'}, {'diagnosis': 'results/global_task_role_fixed_best_20261004_v1/RGBNT100_semantic/DIAGNOSIS.json', 'diagnosis_sha256': '65d0fc24a5d0bb05a84b07b96ec3f7a5b50066638edc30a53324d6824f0518c0', 'target': 'results/global_task_role_fixed_best_20261004_v1/RGBNT100_semantic/query_features.pt', 'bytes': 42185559, 'sha256': '6a2642477ae744eb5b3039e322e533fdf5fed087b137a5c17f3019e1166dbb60'}, {'diagnosis': 'results/global_task_role_fixed_best_20261004_v1/RGBNT100_native/DIAGNOSIS.json', 'diagnosis_sha256': '9e4697dd2f1d5785a40e8e13f77cc79c9fe21c9958474e42b731d8db0ba3d497', 'target': 'results/global_task_role_fixed_best_20261004_v1/RGBNT100_native/query_features.pt', 'bytes': 42199838, 'sha256': '84f03448669ed494eb5716a2ad652b15bde0678a9888aa5394d0fe5f57a95361'}]:
 target=root/r['target'];diagnosis=root/r['diagnosis']
 assert target.resolve().is_relative_to((root/'results/global_task_role_fixed_best_20261004_v1').resolve())
 assert target.name=='query_features.pt' and str(target) not in seal['artifact_sha256']
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
record=dict(status='CONSUMED_HISTORICAL_QUERY_ONLY_RETIREMENT_COMPLETE',at=datetime.now().astimezone().isoformat(),rows=rows,
 retired_files=sum(r['present_before'] for r in rows),retired_bytes=sum(r['bytes'] for r in rows if r['present_before']),
 disk_free_before=before,disk_free_after=after,
 boundary='Only already diagnosed historical query_features caches, not in next187 dependencies. Every formal best, initializer, original evaluation distances, query features, DIAGNOSIS and source retained. No new model, current cache, failure weight, recursive deletion or scientific result change.')
(journal/'RETIREMENT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
