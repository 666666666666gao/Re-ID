from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
qpath=root/'logs/selection_obsolete_role_pool_qualification_20261006_869_r3/QUALIFIED.json'
assert sha(qpath)=='93ff5d26fb824eaa6d8aec39ce36096b95dab1f5068e6827b67d8b71a3d867dd'
q=json.loads(qpath.read_text());assert q['status']=='QUALIFIED_NOT_DELETED'
keep={str(root/'trained-model'/n/'best_map.pth') for n in (
 'visual_update_control_20261001_v1_visual_update_low_lr_global_only_RGBNT100_seed42_full',
 'visual_update_control_20261001_v1_visual_update_low_lr_global_only_MSVR310_seed42_full',
 'shared_private_evidence_20261001_v2_shared_private_coupled_roles_RGBNT100_seed42_full')}
pairs=[x for x in q['targets'] if x['target']['path'] not in keep]
assert len(pairs)==101 and sum(x['target']['bytes'] for x in pairs)==1519366499
seal=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(seal)=='9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
protected=json.loads(seal.read_text())['artifact_sha256'];assert len(protected)==187
launch=root/'logs/signal_selection_reference_launch_20261006_868'
assert json.loads((launch/'EXIT.json').read_text())['exit_code']==1
for name in ('LAUNCH.json','CHILD.json'):
 r=json.loads((launch/name).read_text());p=Path('/proc')/str(r['pid'])
 assert not p.exists() or int((p/'stat').read_text().split()[21])!=r['start_ticks']
retained={x['retained']['path']:x['retained']['sha256'] for x in pairs}
retained.update({x['path']:x['sha256'] for x in q['retained']})
for x in q['targets']:
 if x['target']['path'] in keep:retained[x['target']['path']]=x['target']['sha256']
assert all(sha(p)==d for p,d in retained.items())
for x in pairs:
 t=x['target'];p=Path(t['path']).resolve()
 assert p.is_relative_to((root/'trained-model').resolve()) and p.name=='best_map.pth' and str(p) not in protected and str(p) not in retained
 assert p.stat().st_nlink==1 and p.stat().st_size==t['bytes'] and sha(p)==t['sha256']
 assert sha(p.parent/'official_metrics.json')==t['receipt_sha256'] and sha(p.parent/'training.json')==t['training_sha256']
 assert sha(t['distance_path'])==t['distance_sha256']
folder=root/'logs/selection_obsolete_pool_retirement_20261006_869';assert not folder.exists();folder.mkdir()
before=dict(status='QUALIFIED_FOR_ONCE_RETIREMENT',at=datetime.now().astimezone().isoformat(),qualification_sha256=sha(qpath),targets=pairs,retained_checkpoint_sha256=retained,explicit_reconstruction_dependencies_retained=sorted(keep),free_bytes=shutil.disk_usage(root).free,
 boundary='Storage retirement only; distinct model conditions are not causal comparisons. Text and official distances retained; no author, current selection, protected187 or nondominated winner deleted. Original qualification transport timeout preserved; actual remote qualification succeeded. No NN/evaluation/report replay.')
(folder/'QUALIFIED.json').write_text(json.dumps(before,indent=2)+'\n')
for x in pairs:
 t=x['target'];p=Path(t['path']);p.unlink()
 with (folder/'RETIREMENTS.jsonl').open('a') as f:f.write(json.dumps(dict(path=str(p),sha256=t['sha256'],bytes=t['bytes'],retained=x['retained']['path'],at=datetime.now().astimezone().isoformat()))+'\n')
assert all(not Path(x['target']['path']).exists() for x in pairs)
assert all(Path(p).is_file() for p in protected) and all(Path(p).is_file() for p in retained)
result=dict(status='COMPLETE',at=datetime.now().astimezone().isoformat(),retired=len(pairs),retired_bytes=sum(x['target']['bytes'] for x in pairs),free_before=before['free_bytes'],free_after=shutil.disk_usage(root).free,protected187_full_hash_verified_at=q['at'],protected187_full_hash_replayed_here=False,retained=len(retained),journal=str(folder))
(folder/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
