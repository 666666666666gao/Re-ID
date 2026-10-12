$ErrorActionPreference='Stop'
$taskCode=@'
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
aggregate=root/'logs/region_use_six_closed_20261013_913/ALL_SIX_RESULTS_912.json'
assert sha(aggregate)=='336b2ed5d3962416c70b08896d05a66482f938c103e2caad8c247ab2e5cc7656'
six=json.loads(aggregate.read_text())['rows']
assert len(six)==6
campaigns={
 'original906':root/'logs/region_use_v1_20261012_906/campaign.json',
 'original911':root/'logs/region_use_remaining_20261013_911/campaign.json',
 'accepted912':root/'logs/region_use_remaining_strict_20261013_912/region_use_remaining_20261013_911/campaign.json'}
states={k:json.loads(p.read_text()) for k,p in campaigns.items()}
exits={
 'original911':root/'logs/region_use_remaining_admin911/EXIT.json',
 'accepted912':root/'logs/region_use_remaining_strict_admin912/EXIT.json'}
assert json.loads(exits['original911'].read_text())['exit_code']==1
assert states['original911']['status']=='STOPPED_BEFORE_STAGE_INSUFFICIENT_STORAGE'
assert json.loads(exits['accepted912'].read_text())['exit_code']==0
assert states['accepted912']['status']=='TRAINING_AND_STRICT_EVALUATION_COMPLETE'
pins={str(aggregate):sha(aggregate)}
pins.update({str(p):sha(p) for p in list(campaigns.values())+list(exits.values())})
rows=[]
for item in six:
 candidate=item['candidate'];run=Path(candidate['run_dir'])
 assert run.resolve().is_relative_to(root.resolve())
 receipt_path=run/'official_metrics.json';training_path=run/'training.json'
 receipt=json.loads(receipt_path.read_text());training=json.loads(training_path.read_text())
 assert receipt['status']=='COMPLETE' and receipt['training_epochs']==50
 assert [h['epoch'] for h in training['history']]==list(range(1,51))
 best=max(training['history'],key=lambda h:(h['official_fused']['mAP'],h['epoch']))
 assert best['epoch']==item['best_epoch']==receipt['selected_epoch']
 assert best['official_fused']==receipt['metrics']==item['metrics']
 assert receipt['checkpoint_sha256']==candidate['checkpoint_sha256']
 assert receipt['distance_sha256']==candidate['distance_sha256']
 name='original911' if (item['dataset'],item['comparison'])==('RGBNT100','ambiguity') else 'original906'
 job=next(j for j in states[name]['jobs'] if (j['dataset'],j['comparison'])==(item['dataset'],item['comparison']))
 stage=next(s for s in job['stages'] if s['mode']=='prepare')
 init=Path(stage['command'][stage['command'].index('--initialization')+1])
 assert init.resolve().is_relative_to(root.resolve())
 assert sha(init)==receipt['condition']['initialization_sha256']
 protocol=root/'logs/training_feature_scale_protocols_20261002'/(item['dataset']+'.json')
 assert sha(protocol)==receipt['protocol_sha256']
 records=json.loads(protocol.read_text())['records']
 counts={k:len(records[k]) for k in ('query','gallery')}
 pins.update({str(p):sha(p) for p in (receipt_path,training_path,init,protocol)})
 binary={key:dict(path=str(run/filename),exists=(run/filename).exists(),bytes=(run/filename).stat().st_size,
                 sha256_from_accepted_receipt=candidate[key],bytes_sha256_rechecked=False)
         for key,filename in (('checkpoint_sha256','best_map.pth'),('distance_sha256','official_distances.pt'))}
 rows.append(dict(dataset=item['dataset'],comparison=item['comparison'],run_dir=str(run),
                  initialization=str(init),initializer=training['initializer'],best_epoch=best['epoch'],metrics=receipt['metrics'],
                  checkpoint_sha256=candidate['checkpoint_sha256'],distance_sha256=candidate['distance_sha256'],
                  query_count=counts['query'],gallery_count=counts['gallery'],original_training_campaign=str(campaigns[name]),
                  strict_acceptance_campaign=str(campaigns['accepted912']) if name=='original911' else str(campaigns[name]),
                  original50_history_validated=True,binaries=binary))
payload=sum(5*4*r['query_count']*r['gallery_count'] for r in rows)
required=2*1024**3+64*1024**2+payload
free=shutil.disk_usage(root).free
value=dict(status='PRIVATE_UNADMITTED_SIX_BEST_ADMINISTRATIVE_BINDINGS',at=datetime.now().astimezone().isoformat(),rows=rows,
           text_artifact_sha256=pins,remaining_campaign=str(campaigns['accepted912']),remaining_exit=str(exits['accepted912']),
           original_failed_remaining_campaign=str(campaigns['original911']),original_failed_remaining_exit=str(exits['original911']),
           diagnostic_payload_bytes=payload,storage_required_bytes=required,observed_free_bytes=free,storage_fit_at_observation=free>=required,
           neural_calls=0,optimizer_updates=0,gpu_reads=0,remote_writes=0,admission=False,
           boundary='Text metadata/initialization/protocol counts and binary presence/size only. Current best/distance bytes were not hashed again. Not a full source/artifact execution seal, GPU idle check or NN admission. User next-scope answer remains required before dependent execution.')
print(json.dumps(value))
'@
$output = $taskCode | & ssh -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes -i C:/Users/gb/.ssh/id_ed25519 -p 2026 gaob@172.19.12.138 /usr/bin/python3 -
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
$target='C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/DIAGNOSTIC_BINDINGS_916.json'
if (Test-Path -LiteralPath $target) {throw 'Existing binding result must not be overwritten'}
[System.IO.File]::WriteAllText($target,($output -join "`n")+"`n",[System.Text.UTF8Encoding]::new($false))
$value=Get-Content -LiteralPath $target -Raw -Encoding UTF8 | ConvertFrom-Json
[PSCustomObject]@{status=$value.status;at=$value.at;rows=$value.rows.Count;payload_bytes=$value.diagnostic_payload_bytes;required_bytes=$value.storage_required_bytes;free_bytes=$value.observed_free_bytes;admission=$value.admission;neural_calls=$value.neural_calls} | ConvertTo-Json -Compress
