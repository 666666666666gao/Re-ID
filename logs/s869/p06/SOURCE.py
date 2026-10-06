from pathlib import Path
from datetime import datetime
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion');campaign=root/'logs/signal_selection_reference_v1_20261006_868'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
state=json.loads((campaign/'campaign.json').read_text())
job=next(j for j in state['jobs'] if (j['dataset'],j['selection'],j['phase'])==('RGBNT100','global_only','full'))
assert job['status']=='COMPLETE' and job['exit_code']==0 and all(s['exit_code']==0 for s in job['steps'])
folder=root/'trained-model/signal_selection_reference_v1_20261006_868_full_global_only_RGBNT100'
training=json.loads((folder/'training.json').read_text());receipt=json.loads((folder/'official_metrics.json').read_text())
assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and receipt['status']=='COMPLETE'
assert [h['epoch'] for h in training['history']]==list(range(1,51))
best=max(training['history'],key=lambda h:(h['official_fused']['mAP'],h['epoch']))
assert best['epoch']==training['best_epoch']==receipt['selected_epoch']==job['result']['best_epoch']
assert all(abs(best['official_fused'][k]-v)<1e-5 for k,v in receipt['metrics'].items())
assert receipt['metrics']==job['result']['metrics'] and receipt['independent_upstream_metrics_equal'] and not receipt['reranking']
steps=(folder/'training_steps.jsonl').read_text().splitlines()
batches=(folder/'training_batch_metadata.jsonl').read_text().splitlines()
count=sum(h['steps'] for h in training['history'])
assert len(steps)==len(batches)==count==job['result']['formal_steps']
assert [json.loads(b)['global_step'] for b in batches]==list(range(1,count+1))
assert sha(folder/'official_metrics.json')==job['result']['receipt_sha256']
binding=job['result']['initializer']
assert binding['batch_size']==128 and binding['num_instances']==16 and binding['feature_width']==1536
actual={n:dict(bytes=(folder/n).stat().st_size,sha256=sha(folder/n)) for n in ('best_map.pth','best_epoch_distances.pt','official_distances.pt')}
assert actual['best_map.pth']['sha256']==receipt['checkpoint_sha256']==job['result']['checkpoint_sha256']
assert actual['official_distances.pt']['sha256']==receipt['distance_sha256']==job['result']['distance_sha256']
assert actual['best_epoch_distances.pt']['sha256']==receipt['training_best_distance_sha256']
m0=root/'trained-model/signal_selection_reference_v1_20261006_868_m0_global_only_RGBNT100'
probe=m0/'m0_reload_probe.pth'
retired=next(r for r in map(json.loads,(campaign/'retired_m0.jsonl').read_text().splitlines()) if r['path']==str(probe))
assert not probe.exists() and retired['formal_receipt_sha256']==sha(folder/'official_metrics.json')
source_map=json.loads((root/'refine-logs/signal_selection_reference_v1/SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(source_map)==374 and all(sha(root/n)==s for n,s in source_map.items())
files=[folder/n for n in ('training.json','training_steps.jsonl','training_batch_metadata.jsonl','official_metrics.json')]
files += [campaign/n for n in ('RGBNT100_global_only_train.log','RGBNT100_global_only_evaluate.log','retired_m0.jsonl')]
snapshot=root/'logs/selection_rgb100_global_intake_20261006_868';assert not snapshot.exists();snapshot.mkdir()
(snapshot/'ACCEPTED_JOB.json').write_text(json.dumps(job,indent=2)+'\n');files.append(snapshot/'ACCEPTED_JOB.json')
times={s['mode']:(datetime.fromisoformat(s['completed_at'])-datetime.fromisoformat(s['started_at'])).total_seconds() for s in job['steps']}
print(json.dumps(dict(status='COMPLETE_ORIGINAL_RGB100_GLOBAL_FIRST_STRICT_INTAKE',at=datetime.now().astimezone().isoformat(),dataset='RGBNT100',selection='global_only',training_epochs=50,formal_steps=count,selected_epoch=receipt['selected_epoch'],metrics=receipt['metrics'],actual_artifacts=actual,formal_receipt_sha256=sha(folder/'official_metrics.json'),source_count=374,retired_m0=retired,job=job,history_last_metrics=training['history'][-1]['official_fused'],best_to_last_map_drop=receipt['metrics']['mAP']-training['history'][-1]['official_fused']['mAP'],train_cli_seconds=times['train'],first_strict_cli_seconds=times['evaluate'],loop_with_epoch_evaluation_save_seconds=(datetime.fromisoformat(training['completed_at'])-datetime.fromisoformat(training['started_at'])).total_seconds(),files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},boundary='Read-only original complete50/firststrict/full-gallery receipts and physicalSHA. Actual B128/K16 stepcount fromlogs; notold6559 B64/K8 count. No model/scoring/report replay ornewcapacity/TriFusiongain claim. OtherRGB100arms andoncefinalall9CPUreport still require independentcompletion. AllCMCfollowone mAP-best; singleconsumedofficialseed42, notstableSOTA.')))
