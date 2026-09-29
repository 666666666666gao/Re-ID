# Context/local identity执行跟踪

2026-09-29：固定seed42，三个数据集各五条件；全部full50/官方mAP-best/独立重载。

| 条件 | RGBNT201 | RGBNT100 | MSVR310 |
|---|---|---|---|
| static_none | VERIFIED_COMPLETE E2 72.5957/74.0431 | VERIFIED_COMPLETE E1 85.1254/94.9854 | VERIFIED_COMPLETE E10 52.2626/67.5127 |
| context_none | VERIFIED_COMPLETE E2 72.6087/74.1627 | M0_PASS / TRAINING_35/50 | VERIFIED_COMPLETE E15 52.9824/68.0203 |
| static_local | M0_PASS / TRAINING_19/50 | M0_PASS / TRAINING_4/50 | M0_PASS / TRAINING_10/50 |
| context_local | READY_NOT_RUN | READY_NOT_RUN | READY_NOT_RUN |
| context_global | READY_NOT_RUN | READY_NOT_RUN | READY_NOT_RUN |

15:06实际新5/15正式接受；原M3已全12/12完成。三个trainer活跃，一卡完成端等待父队列接续；READY_NOT_RUN不等于M0或正式结果。

2026-09-29 13:10：五份独立源码实现；CPU合成初始化/查询/八步梯度检查通过，所有五条件原主输出一致；真实M0未运行，15端未启动。前序固定为原M3全12端完成且CPU验收通过，不抢占或提前启动新GPU任务。执行链fresh复核进行中。

2026-09-29 13:20启动前登记：原M3已验收11/12，只剩100 matched/predictor训练。为兑现四卡利用率，新15在空卡接续；旧worker GPU在train→evaluate阶段切换也保留，worker前等待其完成。新15终态前仍要求原12完整CPU验收。模型/入口和当前执行链fresh复核通过；真实生产M0与正式GPU尚未运行，不将旧CPU玩具检查当M0。

2026-09-29 13:30实际四卡训练；新controller2779171启动2026-09-29T13:27:25.681911+08:00，三个static_none真实M0全119/119、baseline不变、reload0；原100 PID2699844保持27/50。其余12端排队，原12／新15终态验收门保持，无中途best采纳。

2026-09-29 14:00：static_none201与MSVR完整50/best/三路完整图库CPU验收，原MSVR接受行未改；context_none201自己的生产M0通过且初始state与静态控制相同。剩余13端继续，不据无辅助控制改辅助ID权重或取消负端。

2026-09-29 14:35：新增201 context_none完整50/best/reload/三路CPU验收，静态→语义条件仅+.0130mAP／+.1196R1、repair1/new0；原两个接受行不改。context_none三个数据集真实M0全部通过且各与静态控制同完整初始化。未选配方、取消条件或新增seed；剩余12正式端继续。

2026-09-29 15:06：原M3全12/12闭合，100四因素首次CPU分析；新static_none100/context_noneMSVR全50/best/三路CPU验收，既有3接受行不改。MSVR语义查询+.7198mAP/+.5076R1，repair24/new21、身份macro略负。static_local201/100各自真实M0通过121/121，原等待guard自动解除，不改冻结源码。剩余10端继续，不选配方。

实际2026-09-29T15:12:55.555311+08:00：四卡自动接续，GPU0 context_none/RGBNT100 35/50，GPU1 static_local/MSVR310 10/50，GPU2 static_local/RGBNT100 4/50，GPU3 static_local/RGBNT201 19/50；全部生产M0通过。新正式仍5/15，剩余10继续。

2026-09-29 17:33完整13/15：collector actual 2026-09-29T17:33:35.975938+08:00，原5条正式接受对象不变，新8端文本training/official/steps归档和receipt SHA核对。剩余RGBNT100 context_local/context_global未完成，未接受中途best。actual progress666为17:33:12而不是原15:31预期；15:31本地观察句柄失效、未写回执，确认后在17:33执行同一observer一次，未重启训练。新最终分析工具待fresh复核，只有完整15/15才运行；partial5 gate拒绝证据保留。17:50 durable一次观察PID3001005，减少轮询。能量诊断source-only审查中，尚无GPU诊断结果，不改原runtime九源。
