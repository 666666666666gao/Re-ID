$ErrorActionPreference='Stop'
$repo='C:\Users\gb\.trifusion_github_publish_22c3bee'
Set-Location -LiteralPath $repo
$utf8=[Text.UTF8Encoding]::new($false)
$now=(Get-Date).ToString('o')
if((git rev-parse HEAD).Trim() -ne 'e75486f07a02465f5f11e2e61afe1467e64b6b14'){throw 'Unexpected HEAD'}
$doc='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
if((Get-FileHash -LiteralPath $doc).Hash.ToLower() -ne '1428e80767977a39b49ab67e4c4129034e6ae477e921d3c39f59f6bd0bc9f1ba'){throw 'Unexpected document'}
$prefix='logs/visual_update_milestone_703_20261001'
$s=Get-Content -LiteralPath "$prefix/SNAPSHOT.json" -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
$intake=Get-Content -LiteralPath "$prefix/INTAKE.json" -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
if($s.formal_parent_complete -lt 1 -or $s.bound_source_count -ne 222 -or $s.analysis_waiter.report_invocations -ne 0){throw 'Unexpected milestone scope'}
$table=@('| Dataset | Condition | Parent status | Child status | GPU | Complete epochs |','|---|---|---|---|---:|---:|')
foreach($j in $s.parent_jobs){
    $e=@($s.endpoints|Where-Object{$_.dataset -eq $j.dataset -and $_.variant -eq $j.variant})
    $child=if($e.Count){$e[0].child_status}else{'-'}
    $g=if($j.status -ne 'PENDING'){$j.gpu}else{'-'}
    $epochs=if($e.Count){$e[0].recorded_complete_epochs}else{'-'}
    $table+=('| {0} | {1} | {2} | {3} | {4} | {5} |' -f $j.dataset,$j.variant,$j.status,$child,$g,$epochs)
}
$formal=@('| Dataset | Condition | best epoch | mAP | Rank-1 | Rank-5 | Rank-10 |','|---|---|---:|---:|---:|---:|---:|')
$accepted=@($s.endpoints|Where-Object{$_.parent_status -eq 'COMPLETE' -and $_.child_status -eq 'COMPLETE'})
foreach($e in $accepted){
    if($e.verification.status -ne 'VERIFIED_COMPLETE' -or @($e.stages|Where-Object{$_.status -ne 'COMPLETE' -or $_.exit_code -ne 0}).Count){throw 'Invalid complete endpoint'}
    $v=$e.verification;$m=$v.metrics
    $formal+=('| {0} | {1} | {2} | {3:F4} | {4:F4} | {5} | {6} |' -f $e.dataset,$e.variant,$v.best_epoch,$m.mAP,$m.'Rank-1',$(if($e.dataset -eq 'RGBNT201'){'{0:F4}' -f $m.'Rank-5'}else{'-'}),$(if($e.dataset -eq 'RGBNT201'){'{0:F4}' -f $m.'Rank-10'}else{'-'}))
}
$pair=@('| Dataset | low_lr roles minus independent global | delta mAP | delta Rank-1 | Full C2 |','|---|---|---:|---:|---|')
foreach($d in @('RGBNT201','RGBNT100','MSVR310')){
    $roles=@($accepted|Where-Object{$_.dataset -eq $d -and $_.variant -eq 'low_lr_roles'})
    $global=@($accepted|Where-Object{$_.dataset -eq $d -and $_.variant -eq 'low_lr_global_only'})
    if($roles.Count -and $global.Count){
        $left=$roles[0].verification.common_initializer_sha256|ConvertTo-Json -Depth 4 -Compress
        $right=$global[0].verification.common_initializer_sha256|ConvertTo-Json -Depth 4 -Compress
        if($left -ne $right){throw 'Common initialization mismatch'}
        $dm=$roles[0].verification.metrics.mAP-$global[0].verification.metrics.mAP
        $dr=$roles[0].verification.metrics.'Rank-1'-$global[0].verification.metrics.'Rank-1'
        $pair+=('| {0} | Complete pair, diagnostic only | {1:F4} | {2:F4} | PENDING_FULL12_AUDIT |' -f $d,$dm,$dr)
    }else{$pair+=('| {0} | Await both complete endpoints | - | - | PENDING_FULL12_AUDIT |' -f $d)}
}
$tracker=@('# Visual-update/readout full50 tracker','',"Recorded $now; one dated sequential snapshot $($s.observed_at). Immutable registration/plan/initialization remain unchanged.",'')+$table+@('',"Parent complete $($s.formal_parent_complete)/12; child verified $($s.child_verified_complete)/12. Official selected metrics only from parent-confirmed complete endpoints.",'')+$formal+@('','Full C1 and C2 remain pending. No replacement of frozen controls, interim-score rescue, new LR, seed or observer. Complete CPU report invocations0; existing waiter owns its one execution after full12.','')+$pair+@('','Goal remains ACTIVE/UNMET; ordinary fine-tuning control is not novelty.')
foreach($p in @('refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER_20261001_703.md','refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER.md')){[IO.File]::WriteAllText((Join-Path $repo $p),($tracker -join "`n")+"`n",$utf8)}
$append=[IO.File]::ReadAllText('C:\Users\gb\.codex_tmp\visual_update_milestone_702_append_20261001.md',$utf8)
$tokens=[ordered]@{RECORDED_AT=$now;SNAPSHOT_AT=$s.observed_at;PARENT_COMPLETE=$s.formal_parent_complete;CHILD_COMPLETE=$s.child_verified_complete;PROGRESS_TABLE=($table -join "`n");FORMAL_TABLE=($formal -join "`n");PAIR_TABLE=($pair -join "`n");REPORT_INVOCATIONS=$s.analysis_waiter.report_invocations;WAITER_STATUS=$s.analysis_waiter.status;RAW_FILES=$intake.files.Count;FREE_BYTES=$s.free_disk_bytes}
foreach($key in $tokens.Keys){$append=$append.Replace('{{'+$key+'}}',[string]$tokens[$key])}
if($append.Contains('{{')){throw 'Unexpanded template'}
$old=[IO.File]::ReadAllBytes((Join-Path $repo $doc));$extra=$utf8.GetBytes($append.TrimEnd([char[]]@("`r","`n"))+"`n")
$all=[byte[]]::new($old.Length+$extra.Length);[Array]::Copy($old,$all,$old.Length);[Array]::Copy($extra,0,$all,$old.Length,$extra.Length)
[IO.File]::WriteAllBytes((Join-Path $repo $doc),$all)
[IO.File]::WriteAllBytes('C:\Users\gb\Desktop\document\TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md',$all)
$entry=@"

| $now | /monitor-experiment | logs/visual_update_milestone_703_20261001/ | implementation | One actual text-only first-endpoint milestone, existing240sec observer and full12 queue preserved |
| $now | /experiment-queue | refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER_20261001_703.md | implementation | Full50 complete endpoints and pending controls recorded; fixed C1/C2 not replaced by partial comparisons |
"@
$stream=[IO.File]::Open((Join-Path $repo 'MANIFEST.md'),[IO.FileMode]::Append);$b=$utf8.GetBytes($entry+"`n");$stream.Write($b,0,$b.Length);$stream.Dispose()
[ordered]@{recorded_at=$now;doc_bytes=$all.Length;doc_sha256=(Get-FileHash -LiteralPath $doc).Hash.ToLower();prefix_bytes_preserved=$old.Length;formal_results=$accepted.Count;full_gates='PENDING'}|ConvertTo-Json
