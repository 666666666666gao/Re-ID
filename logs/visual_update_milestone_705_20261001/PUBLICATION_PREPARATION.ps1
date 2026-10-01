$ErrorActionPreference='Stop'
$repo='C:/Users/gb/.trifusion_github_publish_22c3bee'
Set-Location -LiteralPath $repo
$utf8=[Text.UTF8Encoding]::new($false)
$now=(Get-Date).ToString('o')
if((git rev-parse HEAD).Trim() -ne '4cf144d2194ad6fd3b2a88bdb42c7b9ce7de3408'){throw 'Unexpected publication HEAD'}
$doc='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
if((Get-FileHash -LiteralPath $doc).Hash.ToLower() -ne 'eb55228617f647453880c2e6c8f6211f8401c3adf7ec9ada0d6755c6f449ad64'){throw 'Unexpected document'}
$prefix='logs/visual_update_milestone_705_20261001'
$s=Get-Content -LiteralPath "$prefix/SNAPSHOT.json" -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
$prior=Get-Content -LiteralPath 'logs/visual_update_milestone_704_20261001/SNAPSHOT.json' -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
$intake=Get-Content -LiteralPath "$prefix/INTAKE.json" -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
if($s.bound_source_count -ne 222 -or $s.analysis_waiter.report_invocations -ne 0 -or $s.formal_parent_complete -lt 6){throw 'Unexpected milestone scope'}
$accepted=@($s.endpoints|Where-Object{$_.parent_status -eq 'COMPLETE' -and $_.child_status -eq 'COMPLETE'})
foreach($e in $accepted){if($e.verification.status -ne 'VERIFIED_COMPLETE' -or @($e.stages|Where-Object{$_.status -ne 'COMPLETE' -or $_.exit_code -ne 0}).Count){throw 'Invalid endpoint'}}
$priorKeys=@($prior.endpoints|Where-Object{$_.parent_status -eq 'COMPLETE' -and $_.child_status -eq 'COMPLETE'}|ForEach-Object{$_.dataset+'/'+$_.variant})
$new=@($accepted|Where-Object{($_.dataset+'/'+$_.variant) -notin $priorKeys})
$table=@('| Dataset | Condition | Parent | Child | GPU | Complete epochs |','|---|---|---|---|---:|---:|')
foreach($j in $s.parent_jobs){
    $e=@($s.endpoints|Where-Object{$_.dataset -eq $j.dataset -and $_.variant -eq $j.variant})
    $child=if($e.Count){$e[0].child_status}else{'-'}
    $g=if($j.status -ne 'PENDING'){$j.gpu}else{'-'}
    $epochs=if($e.Count){$e[0].recorded_complete_epochs}else{'-'}
    $table+=('| {0} | {1} | {2} | {3} | {4} | {5} |' -f $j.dataset,$j.variant,$j.status,$child,$g,$epochs)
}
$formal=@('| Dataset | Condition | best epoch | mAP | R1 | R5 | R10 |','|---|---|---:|---:|---:|---:|---:|')
$newFormal=@($formal)
foreach($e in $accepted){
    $v=$e.verification;$m=$v.metrics
    $row='| {0} | {1} | {2} | {3:F4} | {4:F4} | {5} | {6} |' -f $e.dataset,$e.variant,$v.best_epoch,$m.mAP,$m.'Rank-1',$(if($e.dataset -eq 'RGBNT201'){'{0:F4}' -f $m.'Rank-5'}else{'-'}),$(if($e.dataset -eq 'RGBNT201'){'{0:F4}' -f $m.'Rank-10'}else{'-'})
    $formal+=$row
    if(($e.dataset+'/'+$e.variant) -notin $priorKeys){$newFormal+=$row}
}
$c1=@('| Dataset/readout | low_lr minus matched frozen mAP | delta R1 | status |','|---|---:|---:|---|')
$pairRecords=@()
foreach($d in @('RGBNT201','RGBNT100','MSVR310')){
    foreach($readout in @('roles','global_only')){
        $left=@($accepted|Where-Object{$_.dataset -eq $d -and $_.variant -eq ('low_lr_'+$readout)})
        $right=@($accepted|Where-Object{$_.dataset -eq $d -and $_.variant -eq ('frozen_'+$readout)})
        if($left.Count -and $right.Count){
            $lv=$left[0].verification;$rv=$right[0].verification
            if(($lv.common_initializer_sha256|ConvertTo-Json -Depth 4 -Compress) -ne ($rv.common_initializer_sha256|ConvertTo-Json -Depth 4 -Compress)){throw 'Common initializer mismatch'}
            $dm=$lv.metrics.mAP-$rv.metrics.mAP;$dr=$lv.metrics.'Rank-1'-$rv.metrics.'Rank-1'
            $c1+=('| {0}/{1} | {2:F4} | {3:F4} | Complete pair; final full12 audit pending |' -f $d,$readout,$dm,$dr)
            $pairRecords+=[ordered]@{dataset=$d;readout=$readout;delta_map=$dm;delta_r1=$dr;status='COMPLETE_PAIR_FULL12_AUDIT_PENDING'}
        }else{$c1+=('| {0}/{1} | - | - | Pending both complete endpoints |' -f $d,$readout)}
    }
}
$tracker=@('# Visual-update/readout full50 tracker','',"Recorded $now; sequential snapshot $($s.observed_at). Fixed source222, budget, registration and common initialization unchanged.",'')+$table+@('',"Parent complete $($s.formal_parent_complete)/12; child verified $($s.child_verified_complete)/12. One mAP-best checkpoint per complete50-epoch endpoint.",'')+$formal+@('','C1 partial pair comparisons; full C1/C2/report/audit pending.','')+$c1+@('','Existing observer240sec owns one CPU complete report; no restart/new LR/seed/interim-score rescue. GoalACTIVE_UNMET.')
foreach($p in @('refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER_20261001_705.md','refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER.md')){[IO.File]::WriteAllText((Join-Path $repo $p),($tracker -join "`n")+"`n",$utf8)}
$append=@"

## §41.705 第二个完整视觉更新对照与连续训练快照

记录 $now；采用顺序文本快照 $($s.observed_at)，父队列完整 $($s.formal_parent_complete)/12、子端验证 $($s.child_verified_complete)/12。新增正式结果如下；旧五项见§41.704及该时点tracker，未完成端不填中途best。每行均完整50轮、同一个mAP-best权重、严格全state重载、原完整图库和过滤；m0/train/evaluate及worker验证实际退出0。

$($newFormal -join "`n")

已完成的视觉更新减冻结配对如下。两侧使用本次共同FP32存储和匹配初始backbone/neck/classifier，不以旧FP16冻结端补控制。

$($c1 -join "`n")

新MSVR310 frozen_roles为52.3728/67.3435，low_lr_roles为53.0027/68.0203，增加0.6299mAP/0.6768R1；与此前201 roles的+1.3448/+1.1962共同支持这两个数据集在roles读出条件下更新视觉有收益。全局读出对照及完整六项C1仍未完成，不能扩大成全部训练边界均有效。相同low_lr条件下，201/MSVR的roles减独立global_only仍分别只有0.1093/0.0673mAP，低于原C2的0.5要求；不因此改门槛、学习率或种子。普通视觉微调属于训练控制，不是角色新颖性。

MSVR冻结roles的第50轮48.6064/66.4975，较best下降3.7664mAP；冻结201与两数据集的low_lr路径也有best到末轮退化。完整轨迹和实际资源成本最终由预登记CPU报告统一整理，单seed/正式集选best不能当未知分布稳定性证明。共享全局流与私有角色流分离仍是收齐本面板后的候选；本次没有启动新结构或重复旧预测器、地址匹配、V17保护。

证据目录：$prefix/，$($intake.files.Count)份原始文本逐字节/SHA核对，包含实际既有240秒observer快照、父/子状态、训练步与评价回执、同步704证明及本次一次性归档源。完整CPU报告调用次数 $($s.analysis_waiter.report_invocations)，由原observer在全部十二端严格验收后执行一次；未加载模型、距离数组或重放神经计算。快照可用磁盘 $($s.free_disk_bytes)字节。222份活动源、witness、preflight、manifest及报告源未变；原模型、数组与图像留远端。GoalACTIVE/UNMET。
"@
$old=[IO.File]::ReadAllBytes((Join-Path $repo $doc));$extra=$utf8.GetBytes($append.TrimEnd([char[]]@("`r","`n"))+"`n")
$all=[byte[]]::new($old.Length+$extra.Length);[Array]::Copy($old,$all,$old.Length);[Array]::Copy($extra,0,$all,$old.Length,$extra.Length)
[IO.File]::WriteAllBytes((Join-Path $repo $doc),$all)
[IO.File]::WriteAllBytes('C:/Users/gb/Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md',$all)
$entry="`n| $now | /monitor-experiment | $prefix/ | implementation | Second fresh visual-update comparison; full12 queue and source222 unchanged |`n"
$stream=[IO.File]::Open((Join-Path $repo 'MANIFEST.md'),[IO.FileMode]::Append);$b=$utf8.GetBytes($entry);$stream.Write($b,0,$b.Length);$stream.Dispose()
Copy-Item -LiteralPath $PSCommandPath -Destination "$prefix/PUBLICATION_PREPARATION.ps1"
$record=[ordered]@{recorded_at=$now;snapshot_at=$s.observed_at;new_complete_endpoints=@($new|ForEach-Object{$_.dataset+'/'+$_.variant});C1_partial_pairs=$pairRecords;scope='Existing full50 completed endpoint text only, no model/array/report replay';doc_bytes=$all.Length;doc_sha256=(Get-FileHash -LiteralPath $doc).Hash.ToLower();prefix_bytes_preserved=$old.Length;full_gates='PENDING_FULL12_AUDIT'}
[IO.File]::WriteAllText("$prefix/PARTIAL_MILESTONE_DERIVATION.json",(($record|ConvertTo-Json -Depth 8).Replace("`r`n","`n"))+"`n",$utf8)
[IO.File]::WriteAllText('C:/Users/gb/.codex_tmp/visual_update_milestone_705_document_preparation_20261001.json',(($record|ConvertTo-Json -Depth 8).Replace("`r`n","`n"))+"`n",$utf8)
$record|ConvertTo-Json -Depth 8
