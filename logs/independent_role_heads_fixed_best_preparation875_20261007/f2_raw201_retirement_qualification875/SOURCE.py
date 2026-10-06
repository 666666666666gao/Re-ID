from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
import torch
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/independent_role_heads_v1_20261006_873'
assert json.loads((root/'logs/independent_role_heads_launch_20261006_873/EXIT.json').read_text())['exit_code']==0
state=json.loads((campaign/'campaign.json').read_text())
assert state['status']=='COMPLETE' and not (Path('/proc')/str(state['controller_pid'])).exists()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
raw_seal=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(raw_seal)=='9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
protected=set(json.loads(raw_seal.read_text())['artifact_sha256'])
matrix=json.loads((campaign/'accepted_matrix.json').read_text())
protected.update(str(Path(r['run_dir'])/'best_map.pth') for r in matrix['rows'])
protected.add(str(root/'pertrained-model/ViT-B-16.pt'))
rows=[]
for variant in ('raw','normalized'):
 folder=root/'trained-model'/f'training_feature_scale_20261003_v2_full_{variant}_RGBNT201'
 p=folder/'best_map.pth';t=json.loads((folder/'training.json').read_text());receipt_path=folder/'official_metrics.json'
 receipt=json.loads(receipt_path.read_text())
 assert t['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and len(t['history'])==50
 assert receipt['status']=='COMPLETE' and receipt['independent_upstream_metrics_equal'] and not receipt['reranking']
 assert receipt['dataset']=='RGBNT201' and receipt['recipe']==variant
 assert t['history'][receipt['selected_epoch']-1]['official_fused']==receipt['metrics']
 assert p.resolve().is_relative_to((root/'trained-model').resolve()) and str(p) not in protected
 assert sha(p)==receipt['checkpoint_sha256']
 payload=torch.load(p,map_location='cpu',weights_only=True)
 assert payload['schema']=='trifusion-training-feature-scale-v1' and payload['epoch']==receipt['selected_epoch']
 assert payload['metrics']==receipt['metrics']
 arrays={name:sha(folder/name) for name in ('best_epoch_distances.pt','official_distances.pt')}
 assert arrays['best_epoch_distances.pt']==receipt['training_best_distance_sha256']
 assert arrays['official_distances.pt']==receipt['distance_sha256']
 rows.append(dict(variant=variant,path=str(p),sha256=sha(p),bytes=p.stat().st_size,nlink=p.stat().st_nlink,
     metrics=receipt['metrics'],best_epoch=receipt['selected_epoch'],training_sha256=sha(folder/'training.json'),
     receipt_sha256=sha(receipt_path),retained_distance_sha256=arrays))
assert all(rows[0]['metrics'][k]<rows[1]['metrics'][k] for k in rows[0]['metrics'])
assert rows[0]['nlink']==1
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),status='QUALIFIED_UNUSED_CLOSED_F2_RAW201_ALL_FOUR_DOMINATED',
 candidate=rows[0],retained_matched_control=rows[1],protected_paths=len(protected),free_bytes=shutil.disk_usage(root).free,
 raw_seal_sha256=sha(raw_seal),nn_forwards=0,optimizer_updates=0,deleted=0,
 boundary='CPU metadata and SHA qualification only. F2 raw201 fresh public-CLIP training is closed and not a current initializer/teacher/control-seal dependency. Matched normalized F2 best and all text/distance evidence will remain. Original checkpoint replay becomes unavailable if later retired; no old result or receipt is relabelled.')))
