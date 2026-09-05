# V23 experiment tracker

更新时间：2026-09-05T22:42:57.229069+08:00；状态M0_AUDITED_PASS_Q1_RUNNING_FIRST_FOLD_NONNEGATIVE_GATE_UNMET。
训练进度观测时间：2026-09-05T22:37:30.290513+08:00，不是文档写入时间。

| ID | 内容 | 状态 | 证据 |
|---|---|---|---|
| V23-SOURCE | ICPL真实入口与移植边界 | DONE_SOURCE_ONLY | docs/SPECTRAL_ADAPTER_SOURCE_NOTES_2026-09-05.md |
| V23-T0 | 五项远端CUDA合成契约 | DONE_PASS | evidence/trifusion_v23_t0_20260905.json |
| V23-M0 | 完整54前向/116优化步 | DONE_PASS | evidence/trifusion_v23_m0_seed42_9f4a10b.json |
| V23-M0-ARRAYS | 全配对、116步分量/固定门 | DONE_PASS | evidence/trifusion_v23_m0_array_verification_20260905.json |
| V23-M0-FILES | 29全文件SHA/执行Git绑定 | DONE_PASS | evidence/trifusion_v23_m0_file_verification_20260905.json |
| V23-M0-AUDIT | 独立完整M0审计 | COMPLETE_WARN_ENGINEERING_PASS | EXPERIMENT_AUDIT_V23_M0.md/json |
| V23-Q1 | 三fold两端20epoch完整五路检索 | RUNNING_ORIGINAL_PID44684 | evidence/trifusion_v23_progress_20260905_223730.json |
| V23-D1 | 141-fit refit与固定30-dev | NOT_QUALIFIED_NOT_RUN | 首折非负门未满足，完整Q1仍须完成 |

独立M0审计完成：overall/integrity WARN，engineering PASS，
fixed M0 PASS（source-only工程门），scientific NOT_QUALIFIED。
审计使用Node.js v24.13.0标准库对全部116步分量及固定门独立重算；
最大总loss差8.940696727410824e-08，entropy floor=0.57838292104621，
固定第100步excess比0.059681899095254606。它没有执行模型、tensor、图像、
GPU、网络、远端或新优化。七份远端大权重仅由执行器全文件SHA清单支持，
没有被本地审计重新持有/散列。
criterion/mamba/semantic_residual/protocol四文件本地CRLF与执行LF原始字节不同，
仅LF归一化SHA一致；当前latest tracker与执行前登记不同，历史Git blob及版本副本保留。
审计39份输入的原始SHA在归档前再次全部匹配，原始报告/请求/回复与审计时
result/tracker快照保存于trace run12。
审计指出M0结果原句“所有Q1计划原样执行”可能误读为已经完成；
保留审计时原文后，当前M0结果改为明确将按计划继续执行并标注22:07证据时间。
这是文字时态澄清，未改变方案、门限、训练或结果。
M0审计不提供终态Q1、D1/dev、官方或SOTA资格，新增1777536可训练参数及
额外反向计算仍是模态机制解释的混杂因素。

22:37:30.290513+08:00实际只读观察原PID44684及原命令仍运行。
已完成fold0两端各20epoch/580步，正在fold1对照epoch16；
全部日志56个Q1 epoch、1608/3360更新，2/6端点receipt、1/3完整配对fold。
GPU6582MiB使用/17546MiB空闲/100%；exit尚不存在。
partial run_summary最后保存elapsed1628.0644698143005秒，
扣除M0的240.08384251594543秒，首个完整配对约23.13分钟。
据此估计整轮23:10–23:20完成，后续按实际里程碑修正。

这是全部已完成fold0的结果：该fold完整1000gallery/47heldout身份，
190合法query，810记录只从query排除而保留在gallery。
计划三fold合计3126gallery/571query；不能将单折写成完整Q1。
本地仅JSON/NumPy重算首折两端全五路AP/Rank、全部query mask和40条训练history；
指标最大差0，loss分量最大舍入差2.9208202856345622e-08。

| 输出 | 对照 mAP | 候选 mAP | 差值 pp | 对照 R1 | 候选 R1 |
|---|---:|---:|---:|---:|---:|
| baseline_only | 68.767642481 | 68.767642481 | +0.000000000000 | 69.473684211 | 69.473684211 |
| fused | 71.598374938 | 71.598253726 | -0.000121211614 | 72.631578947 | 71.052631579 |
| cnn | 71.166864248 | 72.318663491 | +1.151799242551 | 71.052631579 | 71.052631579 |
| transformer | 70.057173310 | 68.858141806 | -1.199031504211 | 73.684210526 | 70.526315789 |
| mamba | 70.191776363 | 69.916211848 | -0.275564514982 | 68.947368421 | 66.842105263 |

fold0融合精确差值-0.00012121161431366545pp，小于0；
固定“每折融合非负”门在该折不满足，不能因显示舍入接近零改门或算通过。
原过程仍须完成剩余四端，不早停、不选点、不改width/stage/scale/epoch/LR/seed。
当前没有完整聚合、21身份Bootstrap或所有五项终态门；没有D1/dev/official资格。
原始JSON保留首折全部AP/Rank、全部五输出、训练绑定；
日志还保留完整已记录的56epoch，未筛选有利输出或隐藏未配对训练。
