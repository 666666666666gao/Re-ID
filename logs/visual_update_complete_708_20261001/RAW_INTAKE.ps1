$ErrorActionPreference='Stop'
$repo='C:/Users/gb/.trifusion_github_publish_22c3bee'
$archive='C:/Users/gb/.codex_tmp/visual_update_complete_intake_708_20261001.tar.gz'
$receipt='C:/Users/gb/.codex_tmp/visual_update_complete_intake_708_20261001.json'
$dest=Join-Path $repo 'logs/visual_update_complete_708_20261001'
if(Test-Path -LiteralPath $dest){throw 'Complete intake already exists'}
$r=Get-Content -LiteralPath $receipt -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
if((Get-FileHash -LiteralPath $archive).Hash.ToLower() -ne $r.archive_sha256 -or (Get-Item -LiteralPath $archive).Length -ne $r.archive_bytes){throw 'Archive mismatch'}
New-Item -ItemType Directory -Path $dest|Out-Null
tar -xzf $archive -C $dest
if($LASTEXITCODE -ne 0){throw 'Archive extraction failed'}
foreach($entry in $r.files){
    $path=Join-Path (Join-Path $dest 'raw') $entry.path
    if((Get-Item -LiteralPath $path).Length -ne $entry.bytes -or (Get-FileHash -LiteralPath $path).Hash.ToLower() -ne $entry.sha256){throw 'Raw file mismatch'}
}
if((Get-FileHash -LiteralPath (Join-Path $dest 'SNAPSHOT.json')).Hash.ToLower() -ne $r.snapshot_sha256){throw 'Snapshot mismatch'}
$s=Get-Content -LiteralPath (Join-Path $dest 'SNAPSHOT.json') -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
if($s.formal_parent_complete -ne 12 -or $s.child_verified_complete -ne 12 -or $s.analysis_waiter.report_invocations -ne 1 -or $s.analysis_waiter.report_exit_code -ne 0){throw 'Scope mismatch'}
Copy-Item -LiteralPath $receipt -Destination (Join-Path $dest 'INTAKE.json')
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $dest 'RAW_INTAKE.ps1')
$summary=Get-Content -LiteralPath (Join-Path $dest 'raw/results/visual_update_control_complete_20261001/SUMMARY.json') -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
[ordered]@{snapshot_at=$s.observed_at;verified_raw_files=$r.files.Count;accepted=$summary.accepted;registered_visual_update_gate=$summary.registered_visual_update_gate;registered_role_gate=$summary.registered_role_gate;report_actual_completed_at=$s.analysis_waiter.report_completed_at;scope='Existing report preserved once; fresh full-result audit pending'}|ConvertTo-Json
