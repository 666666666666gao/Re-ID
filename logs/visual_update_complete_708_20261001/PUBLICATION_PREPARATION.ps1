$ErrorActionPreference='Stop'
$repo='C:/Users/gb/.trifusion_github_publish_22c3bee'
Set-Location -LiteralPath $repo
$utf8=[Text.UTF8Encoding]::new($false)
if((git rev-parse HEAD).Trim() -ne 'd72b62deb7ef87aef71f690c6a55d4bd707b92ad'){throw 'Unexpected HEAD'}
$doc=Join-Path $repo 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
if((Get-FileHash -LiteralPath $doc).Hash.ToLower() -ne '71816df830851f61d13e19297402fcb0b359919eb546d272c3c0c1d1bbc1f2ba'){throw 'Unexpected document'}
$prefix=Join-Path $repo 'logs/visual_update_complete_708_20261001'
$report=Join-Path $repo 'results/visual_update_control_complete_20261001'
$s=Get-Content -LiteralPath (Join-Path $prefix 'SNAPSHOT.json') -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
$intake=Get-Content -LiteralPath (Join-Path $prefix 'INTAKE.json') -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
$summary=Get-Content -LiteralPath (Join-Path $report 'SUMMARY.json') -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
$audit=Get-Content -LiteralPath (Join-Path $report 'EXPERIMENT_AUDIT.json') -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
if($s.formal_parent_complete -ne 12 -or $s.child_verified_complete -ne 12 -or $summary.accepted -ne 12 -or $summary.rows.Count -ne 12 -or $s.analysis_waiter.report_invocations -ne 1 -or $s.analysis_waiter.report_exit_code -ne 0){throw 'Unexpected complete scope'}
if($audit.review_independence -ne 'same-family' -or $audit.acceptance_status -ne 'provisional' -or $audit.agent_id -ne '/root/audit_visual_update_complete_708'){throw 'Missing actual audit route'}
$now=(Get-Date).ToString('o')
$table=@('| Dataset | Condition | best epoch | mAP | R1 | R5 | R10 |','|---|---|---:|---:|---:|---:|---:|')
foreach($row in $summary.rows){$table+=('| {0} | {1} | {2} | {3:F4} | {4:F4} | {5:F4} | {6:F4} |' -f $row.dataset,$row.variant,$row.best_epoch,$row.metrics.mAP,$row.metrics.'Rank-1',$row.metrics.'Rank-5',$row.metrics.'Rank-10')}
$contrast=@('| Dataset | Visual update: global ΔmAP/ΔR1 | Visual update: roles ΔmAP/ΔR1 | low_lr roles − independent global ΔmAP/ΔR1 |','|---|---:|---:|---:|')
foreach($dataset in @('RGBNT201','RGBNT100','MSVR310')){
 $rows=@($summary.rows|Where-Object{$_.dataset -eq $dataset})
 $fg=$rows|Where-Object{$_.variant -eq 'frozen_global_only'}
 $fr=$rows|Where-Object{$_.variant -eq 'frozen_roles'}
 $lg=$rows|Where-Object{$_.variant -eq 'low_lr_global_only'}
 $lr=$rows|Where-Object{$_.variant -eq 'low_lr_roles'}
 $contrast+=('| {0} | {1:+0.0000;-0.0000;0.0000}/{2:+0.0000;-0.0000;0.0000} | {3:+0.0000;-0.0000;0.0000}/{4:+0.0000;-0.0000;0.0000} | {5:+0.0000;-0.0000;0.0000}/{6:+0.0000;-0.0000;0.0000} |' -f $dataset,($lg.metrics.mAP-$fg.metrics.mAP),($lg.metrics.'Rank-1'-$fg.metrics.'Rank-1'),($lr.metrics.mAP-$fr.metrics.mAP),($lr.metrics.'Rank-1'-$fr.metrics.'Rank-1'),($lr.metrics.mAP-$lg.metrics.mAP),($lr.metrics.'Rank-1'-$lg.metrics.'Rank-1'))
}
$cost=@('| Dataset | Condition | train + per-epoch evaluation seconds | peak allocated GiB | positive Triplet steps / all steps | final − best mAP |','|---|---|---:|---:|---:|---:|')
foreach($row in $summary.rows){$cost+=('| {0} | {1} | {2:F1} | {3:F2} | {4}/{5} | {6:F4} |' -f $row.dataset,$row.variant,$row.training_and_epoch_eval_seconds,($row.peak_training_and_epoch_eval_allocated_bytes/1GB),$row.positive_triplet_steps,$row.logged_steps,$row.final_minus_best_metrics_pp.mAP)}
$tracker=@('# Visual-update/readout full50 tracker','',("Recorded {0}; all12 full50 endpoints verified; original CPU report once exit0 at {1}." -f $now,$s.analysis_waiter.report_completed_at),'')+$table+@('')+$contrast+@('',("Registered visual gate: {0}; registered role gate: {1}; joint gate: {2}." -f $summary.registered_visual_update_gate,$summary.registered_role_gate,$summary.registered_joint_gate),("Actual fresh gpt-6-astra max audit: {0}; same-family/provisional. See results/visual_update_control_complete_20261001/EXPERIMENT_AUDIT.md." -f $audit.verdict),'','No source222, metric, gate, learning-rate, seed or checkpoint selection change. All complete scientific failures preserved. GoalACTIVE_UNMET.')
foreach($name in @('EXPERIMENT_TRACKER_20261001_708.md','EXPERIMENT_TRACKER.md')){[IO.File]::WriteAllText((Join-Path $repo ('refine-logs/visual_update_control_v1/'+$name)),($tracker -join "`n")+"`n",$utf8)}
$append=@"

## §41.708 视觉训练边界十二端全部验收：全局适配受益，角色独立增量仍不足

记录 $now。原队列十二项全部完成50轮，m0/train/evaluate与严格重载均保留实际退出回执；原240秒observer在17:36:41启动唯一CPU报告进程2719272，17:37:10.786035退出0，外层等待器也退出0。接收快照为 $($s.observed_at)，185份原始文件逐字节/SHA归档于logs/visual_update_complete_708_20261001/；原报告四文件完整镜像至results/visual_update_control_complete_20261001/，没有重跑神经前向、训练或CPU报告。

$($table -join "`n")

本面板固定M1第4/8/12层、static global-token、128个Patch内容查询与1536维区域读出，M3及局部ID辅助监督关闭。它是训练边界×读出对照，不是原完整111的新成绩。以上每行所有列来自同一个最高官方fused mAP checkpoint；best是第1/2/10轮并不表示提前停止，全部跑完50轮。四条件视觉存储统一FP32，共同backbone/neck/classifier初始化按字典逐项匹配；low_lr只更新152个视觉张量，固定5e-6；相机及非视觉旧状态保持。global-only独立训练，仍含M1九个适配器的平均写回，关闭角色算子与角色读出；它不是无适配纯CLIP，也不是roles同权重截取global。

$($contrast -join "`n")

独立global-only的视觉更新三集均有正mAP与非负R1，是训练边界的局部正证据。但完整C1要求两种读出都满足，RGBNT100 roles的R1为-0.1166，因此原视觉门为$($summary.registered_visual_update_gate)。C2只使用low_lr下roles相对独立global的三项比较，201/MSVR要求至少0.5mAP；实际约+0.1093、-0.0219、+0.0673，三项均未过原条件，角色门为$($summary.registered_role_gate)，联合门为$($summary.registered_joint_gate)。不根据官方分数修改门槛、学习率、seed或轮次。普通微调不是角色创新，也不支持“所有问题只是冻结主干”。

原报告保留全部六项视觉比较、三项注册角色比较、冻结角色诊断、修复与新增首位错误、身份宏平均AP及固定模型身份bootstrap。低学习率roles相对global的首位修复/新增错误分别为201的3/0、100的1/5、MSVR的8/7；这只是已选模型的真实标签事后解释。bootstrap不是训练多种子，不是未消费官方集的独立泛化检验。RGBNT201修复数较多也不能替代原mAP要求。

$($cost -join "`n")

原trajectories.png/svg显示所有十二项最终mAP均低于所选best，训练loss仍总体下降。延长到50轮已经完成，不能再把后期退化写成“训练还不够”。报告列出的时间包含该阶段训练与逐轮评价，不包括此前ReID底座训练；峰值是初始化后allocated显存，不含初始化瞬时峰值或reserved显存。相同50轮不是相同算力。来源Triplet活动步只描述当前hinge支持，不能换算成AdamW更新份额或某项监督的性能贡献。

新鲜审计实际verdict为$($audit.verdict)，gpt-6-astra/max、独立上下文、同模型家族/provisional；原响应及A–F证据见EXPERIMENT_AUDIT.md/json，私人完整调用trace不提交。审计范围及具体保留项以原响应为准，不把科学门FAIL和结果真实性核查混为一谈。

本次接收器原版错误地断言training.json.status等于COMPLETE，实际十二份状态均为BEST_OFFICIAL_MAP_TRAINING_COMPLETE，17:40:12中止于归档前。原脚本、定时器回执与stderr原样保留；仅修正接收器状态字符串后17:43:49归档成功，实验、权重、原报告与原门槛没有变化。该错误属于证据接收，不是训练失败，不补造重试训练。

下一阶段依据完整面板研究共享全局流与私有角色证据的分离；首先登记一个结构干预并配独立global-only及参数控制，不能同时重写M2对应、M3职责、采样器、视觉学习率和读出。共享/私有分离是待验证假设，不提前声称保住global或十点增益。已完成的M3地址×预测器、V17关系保护、提示carry、全patch/池化与global-token保持封存，不重新换名立项。当前仍是ReID底座上的第二阶段、seed42、官方选best，未获得完整流程多seed、三集强baseline或经协议/资源核实SOTA证据，GoalACTIVE/UNMET。
"@
$old=[IO.File]::ReadAllBytes($doc);$extra=$utf8.GetBytes($append.TrimEnd([char[]]@("`r","`n"))+"`n")
$all=[byte[]]::new($old.Length+$extra.Length);[Array]::Copy($old,$all,$old.Length);[Array]::Copy($extra,0,$all,$old.Length,$extra.Length)
[IO.File]::WriteAllBytes($doc,$all);[IO.File]::WriteAllBytes('C:/Users/gb/Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md',$all)
[IO.File]::AppendAllText((Join-Path $repo 'MANIFEST.md'),("`n| {0} | /experiment-audit | results/visual_update_control_complete_20261001/ | implementation | Full12 matched controls and actual fresh integrity audit; scientific gates retained |`n" -f $now),$utf8)
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $prefix 'PUBLICATION_PREPARATION.ps1')
$record=[ordered]@{recorded_at=$now;accepted=12;report_invocations=1;report_actual_exit_code=0;audit_verdict=$audit.verdict;review_independence='same-family';acceptance_status='provisional';doc_bytes=$all.Length;doc_sha256=(Get-FileHash -LiteralPath $doc).Hash.ToLower();prefix_bytes_preserved=$old.Length;summary_sha256=(Get-FileHash -LiteralPath (Join-Path $report 'SUMMARY.json')).Hash.ToLower();audit_sha256=(Get-FileHash -LiteralPath (Join-Path $report 'EXPERIMENT_AUDIT.md')).Hash.ToLower()}
$json=($record|ConvertTo-Json -Depth 8).Replace("`r`n","`n")+"`n"
[IO.File]::WriteAllText((Join-Path $prefix 'PUBLICATION.json'),$json,$utf8)
[IO.File]::WriteAllText('C:/Users/gb/.codex_tmp/visual_update_complete_708_document_preparation_20261001.json',$json,$utf8)
$record|ConvertTo-Json
