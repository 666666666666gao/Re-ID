from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
active=[]
for directory in Path('/proc').iterdir():
 if directory.name.isdigit() and (directory/'cmdline').is_file():
  command=(directory/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if '/data/gaob/Re-ID/Trifusion/tools/' in command:active.append(dict(pid=directory.name,command=command))
assert not active,active
gpu=subprocess.check_output(['nvidia-smi','-i','0,1','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<500 for line in gpu.splitlines())
protected=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
sources=json.loads((root/'refine-logs/row_mass_role_transport_v2/SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(protected)==187 and len(sources)==366
assert all(sha(n)==d for n,d in protected.items()) and all(sha(root/n)==d for n,d in sources.items())
names=['clean_clip_joint_20261002_v1_clean_clip_global_only_MSVR310_seed42_full','clean_clip_joint_20261002_v1_clean_clip_roles_MSVR310_seed42_full','metric_feature_scale_20261003_v1_full_metric_raw_MSVR310']
rows=[]
for name in names:
 folder=root/'trained-model'/name
 t=json.loads((folder/'training.json').read_text());r=json.loads((folder/'official_metrics.json').read_text())
 assert [x['epoch'] for x in t['history']]==list(range(1,51)) and r['status']=='COMPLETE'
 assert sha(folder/'best_map.pth')==r['checkpoint_sha256'] and sha(folder/'official_distances.pt')==r['distance_sha256']
 assert next(x for x in t['history'] if x['epoch']==t['best_epoch'])['official_fused']==r['metrics']
 assert str(folder/'best_map.pth') not in protected
 rows.append(dict(directory=str(folder),metrics=r['metrics'],checkpoint_sha256=r['checkpoint_sha256'],bytes=(folder/'best_map.pth').stat().st_size,files_sha256={str(p):sha(p) for p in folder.iterdir() if p.is_file()}))
winner=rows[-1]
assert all(all(winner['metrics'][k]>=v for k,v in row['metrics'].items()) and winner['metrics']['mAP']>row['metrics']['mAP'] for row in rows[:-1])
assert not any(n.startswith(tuple(row['directory']+'/' for row in rows[:-1])) for n in protected)
print(json.dumps(dict(status='TWO_CLOSED_WEIGHTS_QUALIFIED_NO_DELETION',free_bytes=shutil.disk_usage(root).free,gpu_memory_only=gpu,rows=rows,sources=366,protected=187)))
