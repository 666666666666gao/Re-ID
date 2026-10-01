$ErrorActionPreference='Stop'
$repo='C:/Users/gb/.trifusion_github_publish_22c3bee'
Set-Location -LiteralPath $repo
$utf8=[Text.UTF8Encoding]::new($false)
if((git rev-parse HEAD).Trim() -ne '7973bedbb91482944b9368fb0e66e5cec787cdba'){throw 'Unexpected HEAD'}
$doc=Join-Path $repo 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
if((Get-FileHash -LiteralPath $doc).Hash.ToLower() -ne '9b7f046286605c741bd4d0dadf32b0ca26f1b0bd289eb00d0a5bf3aca13070e6'){throw 'Unexpected document'}
$prefix=Join-Path $repo 'logs/visual_update_milestone_706_20261001'
$s=Get-Content -LiteralPath (Join-Path $prefix 'SNAPSHOT.json') -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
$intake=Get-Content -LiteralPath (Join-Path $prefix 'INTAKE.json') -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
if($s.formal_parent_complete -ne 8 -or $s.child_verified_complete -ne 8 -or $s.bound_source_count -ne 222 -or $s.analysis_waiter.report_invocations -ne 0){throw 'Unexpected scope'}
$accepted=@($s.endpoints|Where-Object{$_.parent_status -eq 'COMPLETE' -and $_.child_status -eq 'COMPLETE'})
foreach($e in $accepted){if($e.verification.status -ne 'VERIFIED_COMPLETE' -or @($e.stages|Where-Object{$_.status -ne 'COMPLETE' -or $_.exit_code -ne 0}).Count){throw 'Invalid complete endpoint'}}
$byVariant=@{}
foreach($e in @($accepted|Where-Object{$_.dataset -eq 'RGBNT201'})){$byVariant.Add($e.variant,$e.verification)}
if($byVariant.Count -ne 4){throw 'Incomplete201matrix'}
$init=@($byVariant.Values|ForEach-Object{$_.common_initializer_sha256|ConvertTo-Json -Depth 4 -Compress}|Select-Object -Unique)
if($init.Count -ne 1){throw 'Common initialization mismatch'}
$matrix=@('| RGBNT201 condition | best epoch | mAP | R1 | R5 | R10 |','|---|---:|---:|---:|---:|---:|')
foreach($key in @('frozen_global_only','frozen_roles','low_lr_global_only','low_lr_roles')){
    $v=$byVariant[$key];$m=$v.metrics
    $matrix+=('| {0} | {1} | {2:F4} | {3:F4} | {4:F4} | {5:F4} |' -f $key,$v.best_epoch,$m.mAP,$m.'Rank-1',$m.'Rank-5',$m.'Rank-10')
}
$contrasts=[ordered]@{visual_global=@('low_lr_global_only','frozen_global_only');visual_roles=@('low_lr_roles','frozen_roles');role_frozen=@('frozen_roles','frozen_global_only');role_low_lr=@('low_lr_roles','low_lr_global_only')}
$derived=[ordered]@{}
$delta=@('| RGBNT201 comparison | delta mAP | delta R1 | delta R5 | delta R10 |','|---|---:|---:|---:|---:|')
foreach($name in $contrasts.Keys){
    $pair=$contrasts[$name];$left=$byVariant[$pair[0]].metrics;$right=$byVariant[$pair[1]].metrics
    $d=[ordered]@{}
    foreach($metric in @('mAP','Rank-1','Rank-5','Rank-10')){$d[$metric]=$left.$metric-$right.$metric}
    $derived[$name]=$d
    $delta+=('| {0} | {1:F4} | {2:F4} | {3:F4} | {4:F4} |' -f $name,$d.mAP,$d.'Rank-1',$d.'Rank-5',$d.'Rank-10')
}
$interaction=[ordered]@{}
foreach($metric in @('mAP','Rank-1','Rank-5','Rank-10')){$interaction[$metric]=$derived.role_low_lr[$metric]-$derived.role_frozen[$metric]}
$now=(Get-Date).ToString('o')
$progress=@('| Dataset | Condition | Parent | Child | GPU | Complete epochs |','|---|---|---|---|---:|---:|')
foreach($j in $s.parent_jobs){
    $e=@($s.endpoints|Where-Object{$_.dataset -eq $j.dataset -and $_.variant -eq $j.variant})
    $child=if($e.Count){$e[0].child_status}else{'-'}
    $gpu=if($j.status -ne 'PENDING'){$j.gpu}else{'-'}
    $epochs=if($e.Count){$e[0].recorded_complete_epochs}else{'-'}
    $progress+=('| {0} | {1} | {2} | {3} | {4} | {5} |' -f $j.dataset,$j.variant,$j.status,$child,$gpu,$epochs)
}
$tracker=@('# Visual-update/readout full50 tracker','',"Recorded $now; sequential snapshot $($s.observed_at). Source222/registration/commoninitialization unchanged.",'')+$progress+@('','First complete2012x2; every row full50 and one mAP-best checkpoint.','')+$matrix+@('','Single-dataset comparisons; fullC1/C2/report/audit pending.','')+$delta+@('','Global-only retains M1 averaged adapters and removes role operators/readout. Fine-tuning control is not novelty. GoalACTIVE_UNMET; no new LR/seed/configuration from partial scores.')
foreach($name in @('EXPERIMENT_TRACKER_20261001_706.md','EXPERIMENT_TRACKER.md')){[IO.File]::WriteAllText((Join-Path $repo ('refine-logs/visual_update_control_v1/'+$name)),($tracker -join "`n")+"`n",$utf8)}
$append=@"

## §41.706 RGBNT201训练边界2×2全部完成：视觉更新有独立收益，角色增量仍薄

记录 $now，顺序文本快照 $($s.observed_at)。父队列及子端严格验收均8/12；四个未完成端继续既定50轮。新完整端frozen_global_only使RGBNT201四条件全部收齐。每行完整50轮后取各自同一官方mAP-best权重，m0/train/evaluate和worker验证实际退出0，完整state严格重载，完整图库及原camera过滤不变。

$($matrix -join "`n")

四条件共同初始backbone/neck/classifier字典SHA相同，冻结与更新都使用本次统一FP32视觉存储。global_only是独立训练，保留M1九个适配器的共享均值写回，关闭CNN/Transformer/Mamba算子和角色读出；不是零适配纯CLIP，也不是从roles同一权重截取global。以下增量由上面四个实际完整权重计算，不跨列拼值。

$($delta -join "`n")

visual_global/visual_roles分别是对应读出的low_lr减frozen；role_frozen/role_low_lr分别是在相同视觉边界下roles减独立global_only。视觉更新在两种读出都提高约1.25—1.34mAP；角色额外mAP却只有冻结0.0143、更新0.1093，后者未达原登记0.5要求。冻结时角色R1还下降0.1196；更新时R1提高0.3588，但R5下降0.4785。更新与角色的差中差仅+0.0950mAP，不能据此称稳定协作。C1在RGBNT201两读出的单数据集条件满足；完整三数据集C1和完整C2仍待全十二端报告与审核。普通5e-6视觉微调是训练控制，不作为算法新颖性，额外M1/角色的收益不能互相冒充。

新冻结global-only best第2轮，第50轮为69.1212/69.4976/77.3923/81.8182，mAP下降3.3732；此前冻结roles下降3.7116，更新两路下降7.8972/8.4237。四路都有后期退化，单seed、官方集逐轮选点不能确定唯一因果或提供训练稳定性证明。当前独立路线绝对性能仍不足；完整Signal增强的83分不并入本表。

证据目录logs/visual_update_milestone_706_20261001/，$($intake.files.Count)份原始文本逐字节/SHA核对；含原240秒observer快照、父/子状态、完整轨迹和正式回执、同步705证明及一次归档源。快照空闲磁盘$($s.free_disk_bytes)字节。222份活动源、manifest、witness、preflight和报告源未改；完整CPU报告调用次数0，由原observer在十二端严格验收后执行一次，本次没有神经计算或数组重放。

下一步收齐MSVR独立冻结global-only和三个剩余RGBNT100端，统一审核固定C1/C2及成本、身份收益和负翻转，再决定共享全局流/私有角色流分离。已完成的M3地址/预测器、V17保护及旧提示/池化控制不重立项。GoalACTIVE/UNMET。
"@
$old=[IO.File]::ReadAllBytes($doc);$extra=$utf8.GetBytes($append.TrimEnd([char[]]@("`r","`n"))+"`n")
$all=[byte[]]::new($old.Length+$extra.Length);[Array]::Copy($old,$all,$old.Length);[Array]::Copy($extra,0,$all,$old.Length,$extra.Length)
[IO.File]::WriteAllBytes($doc,$all)
[IO.File]::WriteAllBytes('C:/Users/gb/Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md',$all)
$entry="`n| $now | /monitor-experiment | logs/visual_update_milestone_706_20261001/ | implementation | First complete201visual2x2, full50 matched controls, roleincrement belowfixedfloor |`n"
$stream=[IO.File]::Open((Join-Path $repo 'MANIFEST.md'),[IO.FileMode]::Append);$bytes=$utf8.GetBytes($entry);$stream.Write($bytes,0,$bytes.Length);$stream.Dispose()
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $prefix 'PUBLICATION_PREPARATION.ps1')
$record=[ordered]@{recorded_at=$now;snapshot_at=$s.observed_at;matrix=$byVariant;comparisons=$derived;interaction=$interaction;scope='Complete2012x2 text-only derivation; no model/distance/report replay; full12audit pending';doc_bytes=$all.Length;doc_sha256=(Get-FileHash -LiteralPath $doc).Hash.ToLower();prefix_bytes_preserved=$old.Length}
$json=($record|ConvertTo-Json -Depth 12).Replace("`r`n","`n")+"`n"
[IO.File]::WriteAllText((Join-Path $prefix 'PARTIAL_201_COMPLETE_MATRIX.json'),$json,$utf8)
[IO.File]::WriteAllText('C:/Users/gb/.codex_tmp/visual_update_milestone_706_document_preparation_20261001.json',$json,$utf8)
[ordered]@{recorded_at=$now;comparisons=$derived;interaction=$interaction;doc_bytes=$all.Length;doc_sha256=$record.doc_sha256;prefix_bytes_preserved=$old.Length}|ConvertTo-Json -Depth 7
