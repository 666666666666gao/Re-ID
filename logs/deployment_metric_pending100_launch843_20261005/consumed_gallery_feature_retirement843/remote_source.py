from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');rows=[{'path': '/data/gaob/Re-ID/Trifusion/results/native_fixed_best_20261004_811/RGBNT100_semantic/gallery_features.pt', 'bytes': 210914225, 'sha256': 'b83fd5afe3eeccf9e0ff1ce0657a0dc10f2a4a00907f04c929b77b1478d54f3f', 'diagnosis_path': '/data/gaob/Re-ID/Trifusion/results/native_fixed_best_20261004_811/RGBNT100_semantic/DIAGNOSIS.json', 'diagnosis_sha256': 'ad40a3767ce7c6a244a54ec7e97e042de7a76cc7508765d7fc06587c33e8fa76', 'distance_sha256': 'e903940182a93b1bd143fb0b25791bd37bae47a2350477aaad4fb8753071faaa'}, {'path': '/data/gaob/Re-ID/Trifusion/results/native_fixed_best_20261004_811/RGBNT100_native/gallery_features.pt', 'bytes': 210983420, 'sha256': '526c6855263211818686f27f73e63425a38f119428483a209b26c056711c21af', 'diagnosis_path': '/data/gaob/Re-ID/Trifusion/results/native_fixed_best_20261004_811/RGBNT100_native/DIAGNOSIS.json', 'diagnosis_sha256': 'ca76b6a919b69b2d0f5def35b98f63c475de7e9ba40d2b2b050980d046ef33d3', 'distance_sha256': 'd040d45840b16f2943d1cbad66d2de37261626c7094ddd6724666d68e2dd31ba'}, {'path': '/data/gaob/Re-ID/Trifusion/results/role_input_detach_fixed_best_20261004_v1/RGBNT100_semantic/gallery_features.pt', 'bytes': 210914225, 'sha256': '8adbe8182e337f84a18c15b5adc9d75052e120f6e9f1729dba6b6bc09cd29b86', 'diagnosis_path': '/data/gaob/Re-ID/Trifusion/results/role_input_detach_fixed_best_20261004_v1/RGBNT100_semantic/DIAGNOSIS.json', 'diagnosis_sha256': '581815e0275d7230518ab8baccb10564989713cc540197da768818c0114e1a0a', 'distance_sha256': '9c5b5d0eb8a30a09ae38ae187335b1fe2613121ce4e315a7e5889f06434e6ecc'}, {'path': '/data/gaob/Re-ID/Trifusion/results/role_input_detach_fixed_best_20261004_v1/RGBNT100_native/gallery_features.pt', 'bytes': 210983420, 'sha256': '54c35b35aa520ee608d2118edacf080a3fca76f866e6757dd9b87b21774ae771', 'diagnosis_path': '/data/gaob/Re-ID/Trifusion/results/role_input_detach_fixed_best_20261004_v1/RGBNT100_native/DIAGNOSIS.json', 'diagnosis_sha256': '69a549ea997e3cd4c0b9c0421c0ca83a8c77efb8d342ca38a0bc0b5a586c4553', 'distance_sha256': 'cdb077f05ef6ca396fff5572dde8cee8cc55f7c12966f2e5179aa8d697a85d19'}]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=set(json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256'])
for row in rows:
 p=Path(row['path']);assert p.resolve().is_relative_to(root/'results') and p.name=='gallery_features.pt' and str(p) not in protected
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 assert sha(Path(row['diagnosis_path']))==row['diagnosis_sha256']
 assert sha(p.parent/'diagnostic_distances.pt')==row['distance_sha256']
archive=root/'logs/consumed_gallery_feature_retirement_20261005_843';assert not archive.exists();archive.mkdir()
cert=dict(status='FOUR_CONSUMED_HISTORICAL_GALLERY_FEATURE_CACHES_VERIFIED',at=datetime.now().astimezone().isoformat(),rows=rows,
 boundary='Old native/detach fixed-best diagnoses already consumed;current187 inputs exclude these caches. Preserve original diagnostic distances,query features,receipts and all model bests. Old cached gallery inspection retired;no NN replay,power-temperature action or current global-task diagnosis deletion.')
accept=archive/'ACCEPTANCE.json';accept.write_text(json.dumps(cert,indent=2)+'\n');before=shutil.disk_usage(root).free
with (archive/'RETIREMENT.jsonl').open('x') as journal:
 for row in rows:
  Path(row['path']).unlink();journal.write(json.dumps(dict(**row,at=datetime.now().astimezone().isoformat()))+'\n');journal.flush()
assert all(not Path(r['path']).exists() and sha(Path(r['diagnosis_path']))==r['diagnosis_sha256'] for r in rows)
summary=dict(status='FOUR_CONSUMED_GALLERY_CACHES_RETIRED',at=datetime.now().astimezone().isoformat(),retired_bytes=sum(r['bytes'] for r in rows),disk_free_before=before,
 disk_free_after=shutil.disk_usage(root).free,acceptance_sha256=sha(accept))
(archive/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(summary=summary,files={str(p.relative_to(root)):dict(text=p.read_text(),sha256=sha(p)) for p in archive.iterdir()})))
