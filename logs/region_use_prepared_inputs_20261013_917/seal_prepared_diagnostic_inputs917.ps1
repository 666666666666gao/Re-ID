$ErrorActionPreference='Stop'
$taskCode=@'
import hashlib,json,time
from pathlib import Path
from datetime import datetime
root=Path('/data/gaob/Re-ID/Trifusion')
started=time.perf_counter()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):
   h.update(block)
 return h.hexdigest()
old_path=root/'refine-logs/region_use_supervision_v1/INPUT_SEAL.json'
assert sha(old_path)=='d9c56b40c377d658fcc879a8db538a0c585e54219e83c7fd6755411dbaeff7ed'
old=json.loads(old_path.read_text())
assert len(old['source_sha256'])==463 and len(old['artifact_sha256'])==50
binding_path=root/'logs/region_use_diagnostic_contract_20261013_916/DIAGNOSTIC_INPUT_BINDING_916.json'
assert sha(binding_path)=='8c75b85298d48a44e8ea4fd96d46c13c6fc42d12a1ecc98c038f1bc98d702118'
binding=json.loads(binding_path.read_text())
assert len(binding['rows'])==6 and binding['admission'] is False
original_source_checks={}
source_bytes=0
for name,expected in old['source_sha256'].items():
 p=root/name;actual=sha(p);assert actual==expected,name
 original_source_checks[name]=actual;source_bytes+=p.stat().st_size
artifacts=dict(old['artifact_sha256'])
for name,h in binding['text_artifact_sha256'].items():
 if name in artifacts:assert artifacts[name]==h
 artifacts[name]=h
for row in binding['rows']:
 for key,filename in (('checkpoint_sha256','best_map.pth'),('distance_sha256','official_distances.pt')):
  name=str(Path(row['run_dir'])/filename);expected=row[key]
  if name in artifacts:assert artifacts[name]==expected
  artifacts[name]=expected
artifacts[str(binding_path)]=sha(binding_path)
artifacts[str(old_path)]=sha(old_path)
strict_seal=root/'refine-logs/region_use_remaining_strict912/INPUT_SEAL.json'
assert sha(strict_seal)=='7210728e8c32e575396f97af72cacf43bab70cd35277155ee5e87eb4c2345df7'
artifacts[str(strict_seal)]=sha(strict_seal)
artifact_checks={};artifact_bytes=0
for name,expected in artifacts.items():
 p=Path(name);actual=sha(p);assert actual==expected,name
 artifact_checks[name]=actual;artifact_bytes+=p.stat().st_size
staged={}
for name,expected in {
 'tools/diagnose_region_use_best.py':'bc73f6f2d8dc399048dca58ba25d99c059f6cb80bae35e5ab01b130c5b438ff4',
 'tools/queue_region_use_best_diagnosis.py':'8999fcc3b3f80f9a44b0c882462a141af068c6e32bdcd3266cfe3e2e610f2294'}.items():
 archive=root/'logs/region_use_diagnostic_contract_20261013_916/source_reference'/name
 assert sha(archive)==expected
 assert not (root/name).exists()
 staged[name]=dict(archive_path=str(archive),sha256=expected,root_target_present=False)
accepted=json.loads((Path(binding['remaining_campaign'])/'campaign.json').read_text())
terminal=json.loads(Path(binding['remaining_exit']).read_text())
assert accepted['status']=='TRAINING_AND_STRICT_EVALUATION_COMPLETE' and terminal['exit_code']==0
assert accepted['jobs'][0]['status']=='COMPLETE'
failed=json.loads(Path(binding['original_failed_remaining_exit']).read_text())
assert failed['exit_code']==1
prospective_sources=dict(original_source_checks)
prospective_sources.update({name:v['sha256'] for name,v in staged.items()})
value=dict(status='PREPARED_EXACT_INPUT_PACKAGE_NOT_EXECUTION_ADMITTED',schema='trifusion-region-use-fixed-best-decomposition-v1',
 checked_at=datetime.now().astimezone().isoformat(),source_sha256=prospective_sources,artifact_sha256=artifact_checks,
 rows=binding['rows'],remaining_campaign=binding['remaining_campaign'],remaining_exit=binding['remaining_exit'],
 original_failed_remaining_campaign=binding['original_failed_remaining_campaign'],original_failed_remaining_exit=binding['original_failed_remaining_exit'],
 storage_required_bytes=binding['storage_required_bytes'],diagnostic_payload_bytes=binding['diagnostic_payload_bytes'],
 original_root_source_checks=original_source_checks,staged_source_checks=staged,
 scope_counts=dict(root_sources=len(original_source_checks),staged_sources=len(staged),artifacts=len(artifact_checks),
                   original_dependencies=50,current_best_weights=6,current_strict_distances=6),
 bytes_read=dict(root_sources=source_bytes,artifacts=artifact_bytes),elapsed_seconds=time.perf_counter()-started,
 neural_calls=0,optimizer_updates=0,remote_writes=0,gpu_reads=0,execution_admission=False,
 boundary='Actual cold streaming SHA verification of original463 root sources, original50 dependencies and all six current best/strict artifacts. The two diagnostic sources are SHA-checked in source_reference archives; they are absent from root tools. This is a prepared package, not an executed NN gate. GPU idle/current storage must be rechecked, archive sources installed only after next scope is selected, and all prospective source/artifact SHA rechecked before any inference. No training/newweight/reselection/scoring change.')
print(json.dumps(value))
'@
$result = $taskCode | & ssh -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes -i C:/Users/gb/.ssh/id_ed25519 -p 2026 gaob@172.19.12.138 /usr/bin/python3 - 2>&1
$code=$LASTEXITCODE
$work='C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903'
[System.IO.File]::WriteAllText("$work/INPUT_PACKAGE_917.log",($result -join "`n")+"`n",[System.Text.UTF8Encoding]::new($false))
[System.IO.File]::WriteAllText("$work/INPUT_PACKAGE_COMMAND_917.json",([PSCustomObject]@{exit_code=$code;completed_at=[DateTimeOffset]::Now.ToString('o');neural_calls=0;optimizer_updates=0;gpu_reads=0;remote_writes=0}|ConvertTo-Json)+"`n",[System.Text.UTF8Encoding]::new($false))
if ($code -ne 0) {Write-Output $result;exit $code}
$target="$work/PREPARED_DIAGNOSTIC_INPUTS_917.json"
if (Test-Path -LiteralPath $target) {throw 'Do not overwrite an original input-package result'}
[System.IO.File]::WriteAllText($target,($result -join "`n")+"`n",[System.Text.UTF8Encoding]::new($false))
$value=Get-Content -LiteralPath $target -Raw -Encoding UTF8|ConvertFrom-Json
[PSCustomObject]@{status=$value.status;at=$value.checked_at;counts=$value.scope_counts;elapsed_seconds=$value.elapsed_seconds;execution_admission=$value.execution_admission;neural_calls=$value.neural_calls}|ConvertTo-Json -Compress
