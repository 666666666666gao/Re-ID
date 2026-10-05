from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');campaign=root/'logs/semantic_capacity_control_v1_20261005_856'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
accepted=json.loads((campaign/'acceptance/RGBNT201_native.json').read_text())
assert accepted['status']=='FULL50_FIRST_STRICT_AND_M0_VERIFIED_BEFORE_RETIREMENT'
assert all(sha(Path(n))==d for n,d in accepted['artifact_sha256'].items())
assert not Path(accepted['probe']['path']).exists()
training=json.loads((Path(accepted['row']['run_dir'])/'training.json').read_text())
assert len(training['history'])==50
controls=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(controls['artifact_sha256'])==187
current=json.loads((campaign/'initialization/RGBNT201_native.json').read_text())
files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [
 campaign/'acceptance/RGBNT201_native.json',campaign/'probe_retirement.jsonl',
 campaign/'initialization/RGBNT201_native.json',campaign/'initial_forward_pair_RGBNT201.json',
 *campaign.glob('RGBNT201*.log'),*campaign.glob('prepare_RGBNT201.log'),
 *campaign.glob('initial_forward_pair_RGBNT201.log'),
 *[Path(n) for n in accepted['artifact_sha256'] if Path(n).suffix in ('.json','.jsonl','.csv','.log')]]}
caches=[]
for diagnosis in (root/'results').glob('*/*/DIAGNOSIS.json'):
 directory=diagnosis.parent
 for name in ('query_features.pt','gallery_features.pt','diagnostic_distances.pt'):
  p=directory/name
  if p.is_file():
   report=json.loads(diagnosis.read_text())
   if report['status']=='COMPLETE' and 'artifacts' in report and name in report['artifacts']:
    info=report['artifacts'][name]
    assert sha(p)==info['sha256'] and p.stat().st_size==info['bytes']
    caches.append(dict(path=str(p),**info,diagnosis=str(diagnosis),diagnosis_sha256=sha(diagnosis),
                       required_raw187=str(p) in controls['artifact_sha256']))
first=dict(status='FIRST_RGBNT201_COMPLETE_VERIFIED',at=datetime.now().astimezone().isoformat(),
 acceptance=accepted,files=files,initializer=current,formal_steps=sum(r['steps'] for r in training['history']),
 best_to_last_map_drop=accepted['row']['metrics']['mAP']-training['history'][-1]['official_fused']['mAP'],
 history=training['history'],controls=[r for r in controls['rows'] if r['dataset']=='RGBNT201'],
 consumed_cache_inventory=caches,disk_free_bytes=shutil.disk_usage(root).free,
 boundary='Only first accepted endpoint hashes and immutable old cache inventory. No new current NN/runtime/GPU query, NNforward, source edit, baseline rerun or remote checkout.')
print(json.dumps(first))
