$ErrorActionPreference='Stop'
$repo='C:/Users/gb/.trifusion_github_publish_22c3bee'
Set-Location -LiteralPath $repo
$utf8=[Text.UTF8Encoding]::new($false)
if((git rev-parse HEAD).Trim() -ne '64b431556f6b8585668ef5a33b1724138a7d3037'){throw 'Unexpected HEAD'}
$doc=Join-Path $repo 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
if((Get-FileHash -LiteralPath $doc).Hash.ToLower() -ne '9db87afa6eb5d295cdbeb5710652a1b31ab5efea87df5e37193a0acd0f5ccbf0'){throw 'Unexpected document'}
$prefix=Join-Path $repo 'logs/visual_update_milestone_707_20261001'
$s=Get-Content -LiteralPath (Join-Path $prefix 'SNAPSHOT.json') -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
$intake=Get-Content -LiteralPath (Join-Path $prefix 'INTAKE.json') -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
if($s.formal_parent_complete -ne 10 -or $s.child_verified_complete -ne 10 -or $s.bound_source_count -ne 222 -or $s.analysis_waiter.report_invocations -ne 0){throw 'Unexpected scope'}
$accepted=@($s.endpoints|Where-Object{$_.parent_status -eq 'COMPLETE' -and $_.child_status -eq 'COMPLETE'})
foreach($e in $accepted){if($e.verification.status -ne 'VERIFIED_COMPLETE' -or @($e.stages|Where-Object{$_.status -ne 'COMPLETE' -or $_.exit_code -ne 0}).Count){throw 'Invalid complete endpoint'}}
foreach($dataset in @('RGBNT201','MSVR310','RGBNT100')){
    $initial=@($accepted|Where-Object{$_.dataset -eq $dataset}|ForEach-Object{$_.verification.common_initializer_sha256|ConvertTo-Json -Depth 4 -Compress}|Select-Object -Unique)
    if($initial.Count -ne 1){throw 'Common initializer mismatch'}
}
$matrix=[ordered]@{}
foreach($e in @($accepted|Where-Object{$_.dataset -eq 'MSVR310'})){$matrix.Add($e.variant,$e.verification)}
if($matrix.Count -ne 4){throw 'Incomplete MSVR matrix'}
$table=@('| MSVR310 condition | best epoch | mAP | R1 |','|---|---:|---:|---:|')
foreach($key in @('frozen_global_only','frozen_roles','low_lr_global_only','low_lr_roles')){
    $v=$matrix[$key];$table+=('| {0} | {1} | {2:F4} | {3:F4} |' -f $key,$v.best_epoch,$v.metrics.mAP,$v.metrics.'Rank-1')
}
$pairs=[ordered]@{visual_global=@('low_lr_global_only','frozen_global_only');visual_roles=@('low_lr_roles','frozen_roles');role_frozen=@('frozen_roles','frozen_global_only');role_low_lr=@('low_lr_roles','low_lr_global_only')}
$derived=[ordered]@{}
$delta=@('| MSVR310 comparison | delta mAP | delta R1 |','|---|---:|---:|')
foreach($name in $pairs.Keys){
    $pair=$pairs[$name];$d=[ordered]@{}
    foreach($metric in @('mAP','Rank-1','Rank-5','Rank-10')){$d[$metric]=$matrix[$pair[0]].metrics.$metric-$matrix[$pair[1]].metrics.$metric}
    $derived[$name]=$d;$delta+=('| {0} | {1:F4} | {2:F4} |' -f $name,$d.mAP,$d.'Rank-1')
}
$interaction=[ordered]@{}
foreach($metric in @('mAP','Rank-1','Rank-5','Rank-10')){$interaction[$metric]=$derived.role_low_lr[$metric]-$derived.role_frozen[$metric]}
$vehicles=[ordered]@{}
foreach($e in @($accepted|Where-Object{$_.dataset -eq 'RGBNT100'})){$vehicles.Add($e.variant,$e.verification)}
if($vehicles.Count -ne 2 -or -not $vehicles.Contains('frozen_roles') -or -not $vehicles.Contains('low_lr_roles')){throw 'Unexpected RGBNT100 scope'}
$vehicleDelta=[ordered]@{}
foreach($metric in @('mAP','Rank-1','Rank-5','Rank-10')){$vehicleDelta[$metric]=$vehicles.low_lr_roles.metrics.$metric-$vehicles.frozen_roles.metrics.$metric}
$vehicleTable=@('| RGBNT100 roles condition | best epoch | mAP | R1 |','|---|---:|---:|')
foreach($key in @('frozen_roles','low_lr_roles')){$v=$vehicles[$key];$vehicleTable+=('| {0} | {1} | {2:F4} | {3:F4} |' -f $key,$v.best_epoch,$v.metrics.mAP,$v.metrics.'Rank-1')}
$now=(Get-Date).ToString('o')
$progress=@('| Dataset | Condition | Parent | Child | GPU | Complete epochs |','|---|---|---|---|---:|---:|')
foreach($e in $s.endpoints){$progress+=('| {0} | {1} | {2} | {3} | {4} | {5} |' -f $e.dataset,$e.variant,$e.parent_status,$e.child_status,$e.gpu,$e.recorded_complete_epochs)}
$tracker=@('# Visual-update/readout full50 tracker','',"Recorded $now; sequential snapshot $($s.observed_at). Ten complete endpoints, two RGBNT100 global-only controls continue.",'')+$progress+@('','MSVR310 full matched2x2; all rows full50, one official-mAP-best checkpoint.','')+$table+@('')+$delta+@('','RGBNT100 roles matched pair; global-only controls unfinished.','')+$vehicleTable+@('','Visual-update roles delta mAP/R1: '+$vehicleDelta.mAP+'/'+$vehicleDelta.'Rank-1'+'. Original all-pair R1>=0 requirement is not met by this pair; no rule retuning.','Full12 report/audit still pending. Source222/commoninit unchanged. No new neural or report execution; goalACTIVE_UNMET.')
foreach($name in @('EXPERIMENT_TRACKER_20261001_707.md','EXPERIMENT_TRACKER.md')){[IO.File]::WriteAllText((Join-Path $repo ('refine-logs/visual_update_control_v1/'+$name)),($tracker -join "`n")+"`n",$utf8)}
$append=@"

## §41.707 MSVR310训练边界2×2收齐，RGBNT100角色配对出现mAP与首位分歧

记录 $now；既定一次归档快照 $($s.observed_at)，父队列和子端严格验收均10/12。新完整端为MSVR310 frozen_global_only与RGBNT100 frozen_roles。165份原始文本逐字节核对，见logs/visual_update_milestone_707_20261001/。所有已完成端各自m0/train/evaluate实际退出0，完整50轮后按同一官方mAP-best权重严格重载，正式标签、完整图库和MSVR同身份同时间段过滤不变。

$($table -join "`n")

$($delta -join "`n")

MSVR两种读出都受益于视觉更新，分别约+0.9874与+0.6299mAP；但在相同视觉边界下，角色增量由冻结的+0.4248mAP缩小为更新的+0.0673，后者仍低于固定0.5要求。角色×视觉交互约-0.3575mAP。不能将视觉更新收益归给角色协作，也不能凭两个完整数据集宣布跨数据集稳定。独立global-only保留M1均值写回适配，关闭角色算子和读出，不是从roles同权重取global。

$($vehicleTable -join "`n")

RGBNT100两端均选中第1轮；low_lr减frozen为+0.2579mAP、-0.1166R1，R5/R10分别+0.0583/+0.2915。这个完整配对没有满足原C1的R1不得下降要求；保留原规则，等剩余global-only两端、统一CPU报告与完整审核，不通过修改阈值、epoch、学习率或种子补救。冻结roles6559步Triplet全部0，更新roles39步为正；这只描述来源hinge支持，不是实际AdamW更新份额或唯一性能原因。

新MSVR冻结global-only第50轮48.4215/64.1286，相对best下降3.5265mAP；新RGBNT100冻结roles末轮81.8050/93.7609，相对best下降3.2951mAP。更新roles100末轮82.1469/94.3440，也有退化。训练轮数已经完整，不能再把后期检索下降解释为尚未跑满。所有结果仍是一个角色阶段seed42、已消费官方集逐轮选点；不是完整流程多种子或未消费测试证据。

剩余RGBNT100 low_lr_global_only和frozen_global_only快照分别45/50与39/50，预计剩余约1042/1153秒，仅为日程估计；不填中途best。磁盘快照84309110784字节空闲，必要权重/数组留远端供最终审核。222份活动源、共有初始化、preflight和报告源未改。完整CPU报告调用次数0，由原240秒observer待全12验收后执行一次；本次只归档已有文本并计算配对算术，不运行模型、重放数组或新增实验。

RGBNT201完整矩阵已见§41.706；两数据集现在都显示视觉更新效应明显大于同边界角色增量。下一步完成最后两端和全12报告/审核，再决定共享全局流与私有角色证据分离；旧M3地址/预测器、V17保护及提示/池化不重立项。GoalACTIVE/UNMET，尚无三数据集baseline/SOTA达成证据。
"@
$old=[IO.File]::ReadAllBytes($doc);$extra=$utf8.GetBytes($append.TrimEnd([char[]]@("`r","`n"))+"`n")
$all=[byte[]]::new($old.Length+$extra.Length);[Array]::Copy($old,$all,$old.Length);[Array]::Copy($extra,0,$all,$old.Length,$extra.Length)
[IO.File]::WriteAllBytes($doc,$all);[IO.File]::WriteAllBytes('C:/Users/gb/Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md',$all)
$entry="`n| $now | /monitor-experiment | logs/visual_update_milestone_707_20261001/ | implementation | CompleteMSVR2x2 and RGBNT100roles pair, ten accepted endpoints |`n"
$stream=[IO.File]::Open((Join-Path $repo 'MANIFEST.md'),[IO.FileMode]::Append);$bytes=$utf8.GetBytes($entry);$stream.Write($bytes,0,$bytes.Length);$stream.Dispose()
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $prefix 'PUBLICATION_PREPARATION.ps1')
$record=[ordered]@{recorded_at=$now;snapshot_at=$s.observed_at;matrix_msvr=$matrix;comparisons_msvr=$derived;interaction_msvr=$interaction;rgbnt100_roles_pair=$vehicles;rgbnt100_roles_visual_delta=$vehicleDelta;scope='Ten full50 accepted; text-only paired derivation, complete12report/audit pending';doc_bytes=$all.Length;doc_sha256=(Get-FileHash -LiteralPath $doc).Hash.ToLower();prefix_bytes_preserved=$old.Length}
$json=($record|ConvertTo-Json -Depth 12).Replace("`r`n","`n")+"`n"
[IO.File]::WriteAllText((Join-Path $prefix 'PARTIAL_MSVR_AND_RGBNT100_ROLES.json'),$json,$utf8)
[IO.File]::WriteAllText('C:/Users/gb/.codex_tmp/visual_update_milestone_707_document_preparation_20261001.json',$json,$utf8)
[ordered]@{recorded_at=$now;comparisons_msvr=$derived;interaction_msvr=$interaction;rgbnt100_roles_visual_delta=$vehicleDelta;doc_bytes=$all.Length;doc_sha256=$record.doc_sha256;prefix_bytes_preserved=$old.Length}|ConvertTo-Json -Depth 7
