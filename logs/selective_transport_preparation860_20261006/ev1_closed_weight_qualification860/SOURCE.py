from pathlib import Path
import json,hashlib,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
target=root/'trained-model/semantic_native_evidence_20261002_v1_clean_clip_combined_RGBNT201_seed42_full'
winner=root/'trained-model/region_reconstruction_v1_20261005_858_full_native_RGBNT201'
expected={'target_weight':'ae9b973c1aeb30c8d7d18c4518d5a5359b61ec1fb6ce20f48713bdc2efc13f55',
'target_receipt':'a9289db2d79f3b9df2e4500ec378a1a0aa0191cd54abcc96b3797d8268a37df8',
'target_distance':'7bac758006388865ab60a2284a95c2da436e2023895b7fd96622761482dc4c83',
'winner_weight':'9b0b9da90e0ac5b231e73b95d712ed827d88cd0860f220bd2b412636e56062eb',
'winner_receipt':'61ad4da9428536a8560b73ec9e52e3363b18abf89146ed39582edac7b4200b41'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
sources=json.loads((root/'logs/region_reconstruction_v1_20261005_858/manifest.json').read_text())['source_sha256']
protected=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
assert len(sources)==354 and len(protected)==187
assert all(sha(root/n)==d for n,d in sources.items())
assert all(sha(Path(n))==d for n,d in protected.items())
path=target/'best_map.pth'
assert path.resolve().is_relative_to((root/'trained-model').resolve())
assert str(path) not in protected and str(path.relative_to(root)) not in protected
assert str(path.relative_to(root)) not in sources
rows={}
for label,folder in [('target',target),('winner',winner)]:
 training=json.loads((folder/'training.json').read_text())
 receipt=json.loads((folder/'official_metrics.json').read_text())
 assert [r['epoch'] for r in training['history']]==list(range(1,51))
 assert receipt['status']=='COMPLETE'
 assert sha(folder/'best_map.pth')==receipt['checkpoint_sha256']==expected[label+'_weight']
 assert sha(folder/'official_metrics.json')==expected[label+'_receipt']
 assert sha(folder/'official_distances.pt')==receipt['distance_sha256']
 if label=='target':assert receipt['distance_sha256']==expected['target_distance']
 rows[label]=dict(directory=str(folder),best_epoch=training['best_epoch'],metrics=receipt['metrics'],
  artifact_sha256={str(folder/n):sha(folder/n) for n in ('training.json','official_metrics.json','official_distances.pt','best_map.pth')},
  weight_bytes=(folder/'best_map.pth').stat().st_size)
assert all(rows['winner']['metrics'][k]>=v for k,v in rows['target']['metrics'].items())
assert rows['winner']['metrics']['mAP']>rows['target']['metrics']['mAP']
print(json.dumps(dict(status='CLOSED_EV1_201_WEIGHT_QUALIFIED_NO_MUTATION',rows=rows,
 free_bytes=shutil.disk_usage(root).free,source_count=len(sources),protected_count=len(protected),
 boundary='Self-trained unsupported EV1 combined201; all four metrics dominated by retained region201mean. Full50 and receipt/actual weight/distance verified. No new-method dependency. Original arrays/text/history retained; deletion will retire this binary replay path.')))