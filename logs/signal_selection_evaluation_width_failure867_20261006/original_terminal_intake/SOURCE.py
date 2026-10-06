from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/signal_selection_reference_v1_20261006_866'
launch=root/'logs/signal_selection_reference_launch_20261006_866'
state=json.loads((campaign/'campaign.json').read_text())
assert json.loads((launch/'EXIT.json').read_text())['exit_code']==1 and state['status']=='FAILED'
assert state['failed_job']['phase']=='full' and state['failed_job']['selection']=='masked'
for step in [s for j in state['jobs'] for s in j.get('steps',[])]+state['preparation']:
 assert not Path('/proc',str(step['pid'])).exists(),step
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
sources=json.loads((campaign/'manifest.json').read_text())['source_sha256']
assert len(sources)==372 and all(sha(root/n)==d for n,d in sources.items())
folders=[campaign,launch,*sorted((root/'trained-model').glob(campaign.name+'_*'))]
files=[p for folder in folders for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.txt','.log','.md','.json','.jsonl')]
rows=[]
for folder in folders[2:]:
 t=json.loads((folder/'training.json').read_text())
 steps=[json.loads(line) for line in (folder/'training_steps.jsonl').read_text().splitlines()]
 batches=[json.loads(line) for line in (folder/'training_batch_metadata.jsonl').read_text().splitlines()]
 row=dict(run_dir=str(folder),status=t['status'],history=t['history'],completed_optimizer_steps=len(steps),recorded_batches=len(batches),m0=t.get('m0'),selection_reference=t.get('selection_reference'))
 if (folder/'official_metrics.json').exists():
  r=json.loads((folder/'official_metrics.json').read_text())
  assert r['status']=='COMPLETE' and len(t['history'])==50
  assert sha(folder/'best_map.pth')==r['checkpoint_sha256'] and sha(folder/'official_distances.pt')==r['distance_sha256']
  row['formal']=r
 rows.append(row)
failed=state['failed_job'];log=campaign/f'{failed["dataset"]}_{failed["selection"]}_{failed["steps"][-1]["mode"]}.log'
result=dict(status='ONE_FORMAL_ACCEPTED_THEN_MASKED_FULL_FAILED',at=datetime.now().astimezone().isoformat(),campaign=state,rows=rows,failed_log=str(log),failed_log_tail=log.read_text()[-12000:],source_sha256=sources,free_bytes=shutil.disk_usage(root).free,text_files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},boundary='Terminal original handles checked; no NN or report replay, weights and arrays stayremote. Globalformal best distinct fromlast; failedmasked no formal score.')
print(json.dumps(result))
