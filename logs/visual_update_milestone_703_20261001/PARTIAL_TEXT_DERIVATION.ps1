$ErrorActionPreference='Stop'
$repo='C:\Users\gb\.trifusion_github_publish_22c3bee'
$prefix='logs/visual_update_milestone_703_20261001'
$utf8=[Text.UTF8Encoding]::new($false)
$snapshotPath=Join-Path $repo "$prefix/SNAPSHOT.json"
$s=Get-Content -LiteralPath $snapshotPath -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
$rows=@()
foreach($e in @($s.endpoints|Where-Object{$_.parent_status -eq 'COMPLETE'})){
    if($e.child_status -ne 'COMPLETE' -or $e.verification.status -ne 'VERIFIED_COMPLETE'){throw 'Incomplete endpoint'}
    $run=$e.stages[-1].output_dir.Replace('/data/gaob/Re-ID/Trifusion/','')
    $p=Join-Path "$repo/$prefix/raw" "$run/training.json"
    $t=Get-Content -LiteralPath $p -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
    if($t.status -ne 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE' -or $t.history.Count -ne 50){throw 'Incomplete trajectory'}
    $best=@($t.history|Where-Object{$_.epoch -eq $t.best_epoch})[0];$end=$t.history[-1]
    $rows+=[ordered]@{dataset=$e.dataset;variant=$e.variant;best_epoch=$t.best_epoch;best_map=$best.official_fused.mAP;best_r1=$best.official_fused.'Rank-1';epoch50_map=$end.official_fused.mAP;epoch50_r1=$end.official_fused.'Rank-1';best_to_final_map=$end.official_fused.mAP-$best.official_fused.mAP;loss_at_best=$best.mean_loss;loss_at_epoch50=$end.mean_loss;training_json_sha256=(Get-FileHash -LiteralPath $p).Hash.ToLower()}
}
$pairs=@()
foreach($d in @('RGBNT201','RGBNT100','MSVR310')){
    $r=@($rows|Where-Object{$_.dataset -eq $d -and $_.variant -eq 'low_lr_roles'})
    $g=@($rows|Where-Object{$_.dataset -eq $d -and $_.variant -eq 'low_lr_global_only'})
    if($r.Count -and $g.Count){
        $dm=$r[0].best_map-$g[0].best_map;$dr=$r[0].best_r1-$g[0].best_r1
        $floor=if($d -eq 'RGBNT100'){0.0}else{0.5}
        $pairs+=[ordered]@{dataset=$d;delta_map=$dm;delta_r1=$dr;registered_map_floor=$floor;local_condition_met=($dm -gt 0 -and $dm -ge $floor -and $dr -ge 0);status='COMPLETE_PAIR_DIAGNOSTIC_FINAL_FULL12_AUDIT_PENDING'}
    }else{$pairs+=[ordered]@{dataset=$d;status='PENDING_BOTH_COMPLETE_ENDPOINTS'}}
}
$record=[ordered]@{derived_at=(Get-Date).ToString('o');snapshot_at=$s.observed_at;snapshot_sha256=(Get-FileHash -LiteralPath $snapshotPath).Hash.ToLower();source='Archived completed full50 JSON logs only';scope='No complete report main, model, distance or inference replay; no configuration action or early cancellation';full12_report_invocations=0;rows=$rows;low_lr_role_pairs=$pairs;all_cross_dataset_gates='Final complete12 audit/report pending; completed201/MSVRpairs do not meet the original .5mAP condition'}
$out=Join-Path $repo "$prefix/PARTIAL_COMPLETED_TEXT_DIAGNOSTICS.json"
if(Test-Path -LiteralPath $out){throw 'Derived text diagnostics already exist'}
[IO.File]::WriteAllText($out,($record|ConvertTo-Json -Depth 8)+"`n",$utf8)
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $repo "$prefix/PARTIAL_TEXT_DERIVATION.ps1")
Copy-Item -LiteralPath 'C:\Users\gb\.codex_tmp\visual_update_source_decision_notes_703_20261001.md' -Destination (Join-Path $repo "$prefix/SOURCE_DECISION_NOTES.md")
[ordered]@{rows=$rows;pairs=$pairs;output_sha256=(Get-FileHash -LiteralPath $out).Hash.ToLower()}|ConvertTo-Json -Depth 6
