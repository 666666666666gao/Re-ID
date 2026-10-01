$ErrorActionPreference='Stop'
$repo='C:/Users/gb/.trifusion_github_publish_22c3bee'
Set-Location -LiteralPath $repo
$utf8=[Text.UTF8Encoding]::new($false)
$now=(Get-Date).ToString('o')
if((git rev-parse HEAD).Trim() -ne '02b22ba43fbc1ae274fc942fe12b59b03f5f3b99'){throw 'Unexpected HEAD'}
$doc='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
if((Get-FileHash -LiteralPath $doc).Hash.ToLower() -ne '84d6552f717fc6e6c70398e5c60df7e72e9e1512427febe577d71996bfe956ce'){throw 'Unexpected document'}
$prefix='logs/visual_update_milestone_704_20261001'
$s=Get-Content -LiteralPath "$prefix/SNAPSHOT.json" -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
if($s.formal_parent_complete -ne 5 -or $s.child_verified_complete -ne 5 -or $s.bound_source_count -ne 222 -or $s.analysis_waiter.report_invocations -ne 0){throw 'Unexpected scope'}
$accepted=@($s.endpoints|Where-Object{$_.parent_status -eq 'COMPLETE' -and $_.child_status -eq 'COMPLETE'})
foreach($e in $accepted){if($e.verification.status -ne 'VERIFIED_COMPLETE' -or @($e.stages|Where-Object{$_.status -ne 'COMPLETE' -or $_.exit_code -ne 0}).Count){throw 'Invalid endpoint'}}
$low=@($accepted|Where-Object{$_.dataset -eq 'RGBNT201' -and $_.variant -eq 'low_lr_roles'})[0]
$frozen=@($accepted|Where-Object{$_.dataset -eq 'RGBNT201' -and $_.variant -eq 'frozen_roles'})[0]
if(($low.verification.common_initializer_sha256|ConvertTo-Json -Depth 4 -Compress) -ne ($frozen.verification.common_initializer_sha256|ConvertTo-Json -Depth 4 -Compress)){throw 'C1 initialization mismatch'}
$newresult=@('| RGBNT201 condition | best epoch | mAP | R1 | R5 | R10 |','|---|---:|---:|---:|---:|---:|')
foreach($e in @($frozen,$low)){
    $v=$e.verification;$m=$v.metrics
    $newresult+=('| {0} | {1} | {2:F4} | {3:F4} | {4:F4} | {5:F4} |' -f $e.variant,$v.best_epoch,$m.mAP,$m.'Rank-1',$m.'Rank-5',$m.'Rank-10')
}
$m1=$low.verification.metrics;$m0=$frozen.verification.metrics
$c1=@('| low_lr_roles minus frozen_roles | delta mAP | delta R1 | delta R5 | delta R10 |','|---|---:|---:|---:|---:|',('| RGBNT201 complete pair | {0:F4} | {1:F4} | {2:F4} | {3:F4} |' -f ($m1.mAP-$m0.mAP),($m1.'Rank-1'-$m0.'Rank-1'),($m1.'Rank-5'-$m0.'Rank-5'),($m1.'Rank-10'-$m0.'Rank-10')))
$table=@('| Dataset | Condition | Parent status | Child status | GPU | Complete epochs |','|---|---|---|---|---:|---:|')
foreach($j in $s.parent_jobs){
    $e=@($s.endpoints|Where-Object{$_.dataset -eq $j.dataset -and $_.variant -eq $j.variant})
    $child=if($e.Count){$e[0].child_status}else{'-'}
    $g=if($j.status -ne 'PENDING'){$j.gpu}else{'-'}
    $epochs=if($e.Count){$e[0].recorded_complete_epochs}else{'-'}
    $table+=('| {0} | {1} | {2} | {3} | {4} | {5} |' -f $j.dataset,$j.variant,$j.status,$child,$g,$epochs)
}
$formal=@('| Dataset | Condition | best epoch | mAP | Rank-1 | Rank-5 | Rank-10 |','|---|---|---:|---:|---:|---:|---:|')
foreach($e in $accepted){
    $v=$e.verification;$m=$v.metrics
    $formal+=('| {0} | {1} | {2} | {3:F4} | {4:F4} | {5} | {6} |' -f $e.dataset,$e.variant,$v.best_epoch,$m.mAP,$m.'Rank-1',$(if($e.dataset -eq 'RGBNT201'){'{0:F4}' -f $m.'Rank-5'}else{'-'}),$(if($e.dataset -eq 'RGBNT201'){'{0:F4}' -f $m.'Rank-10'}else{'-'}))
}
$tracker=@('# Visual-update/readout full50 tracker','',"Recorded $now; sequential snapshot $($s.observed_at). Registered sources/plan/initialization unchanged.",'')+$table+@('',"Parent/child verified complete5/12. All complete entries have actual three stage exit0 and full50.",'')+$formal+@('','First complete visual-update comparison; full C1/C2 and audit pending.','')+$c1+@('','Existing240sec observer owns the one full12 CPU report. No new LR/seed/configuration or interim-score rescue. GoalACTIVE_UNMET.')
foreach($p in @('refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER_20261001_704.md','refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER.md')){[IO.File]::WriteAllText((Join-Path $repo $p),($tracker -join "`n")+"`n",$utf8)}
$append=[IO.File]::ReadAllText('C:/Users/gb/.codex_tmp/visual_update_milestone_704_append_20261001.md',$utf8)
$tokens=[ordered]@{RECORDED_AT=$now;SNAPSHOT_AT=$s.observed_at;PARENT_COMPLETE=$s.formal_parent_complete;NEW_RESULT_TABLE=($newresult -join "`n");NEW_C1_TABLE=($c1 -join "`n");REPORT_INVOCATIONS=$s.analysis_waiter.report_invocations;FREE_BYTES=$s.free_disk_bytes}
foreach($key in $tokens.Keys){$append=$append.Replace('{{'+$key+'}}',[string]$tokens[$key])}
if($append.Contains('{{')){throw 'Unexpanded template'}
$old=[IO.File]::ReadAllBytes((Join-Path $repo $doc));$extra=$utf8.GetBytes($append.TrimEnd([char[]]@("`r","`n"))+"`n")
$all=[byte[]]::new($old.Length+$extra.Length);[Array]::Copy($old,$all,$old.Length);[Array]::Copy($extra,0,$all,$old.Length,$extra.Length)
[IO.File]::WriteAllBytes((Join-Path $repo $doc),$all)
[IO.File]::WriteAllBytes('C:/Users/gb/Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md',$all)
$entry="`n| $now | /monitor-experiment | $prefix/ | implementation | First complete201visual-update pair, fresh FP32 frozen control, sources222 unchanged |`n"
$stream=[IO.File]::Open((Join-Path $repo 'MANIFEST.md'),[IO.FileMode]::Append);$b=$utf8.GetBytes($entry);$stream.Write($b,0,$b.Length);$stream.Dispose()
Copy-Item -LiteralPath $PSCommandPath -Destination "$prefix/PUBLICATION_PREPARATION.ps1"
Copy-Item -LiteralPath 'C:/Users/gb/.codex_tmp/visual_update_milestone_704_append_20261001.md' -Destination "$prefix/APPEND_TEMPLATE.md"
$record=[ordered]@{recorded_at=$now;doc_bytes=$all.Length;doc_sha256=(Get-FileHash -LiteralPath $doc).Hash.ToLower();prefix_bytes_preserved=$old.Length;formal_results=$accepted.Count;first201_C1_delta_map=$m1.mAP-$m0.mAP;full_gates='PENDING'}
[IO.File]::WriteAllText('C:/Users/gb/.codex_tmp/visual_update_milestone_704_document_preparation_20261001.json',($record|ConvertTo-Json)+"`n",$utf8)
$record|ConvertTo-Json
