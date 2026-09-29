# Context/local identity执行跟踪

2026-09-29：固定seed42，三个数据集各五条件；全部full50/官方mAP-best/独立重载。

| 条件 | RGBNT201 | RGBNT100 | MSVR310 |
|---|---|---|---|
| static_none | VERIFIED_COMPLETE E2 72.5957/74.0431 | M0_PASS / TRAINING_16/50 | VERIFIED_COMPLETE E10 52.2626/67.5127 |
| context_none | M0_PASS / TRAINING_5/50 | M0_RUNNING | READY_NOT_RUN |
| static_local | READY_NOT_RUN | READY_NOT_RUN | READY_NOT_RUN |
| context_local | READY_NOT_RUN | READY_NOT_RUN | READY_NOT_RUN |
| context_global | READY_NOT_RUN | READY_NOT_RUN | READY_NOT_RUN |

14:00实际2正式终点／3running／10pending；其余READY_NOT_RUN不等于M0通过或正式成绩。原M3最后100继续35/50，尚未全12终态。

2026-09-29 13:10：五份独立源码实现；CPU合成初始化/查询/八步梯度检查通过，所有五条件原主输出一致；真实M0未运行，15端未启动。前序固定为原M3全12端完成且CPU验收通过，不抢占或提前启动新GPU任务。执行链fresh复核进行中。

2026-09-29 13:20启动前登记：原M3已验收11/12，只剩100 matched/predictor训练。为兑现四卡利用率，新15在空卡接续；旧worker GPU在train→evaluate阶段切换也保留，worker前等待其完成。新15终态前仍要求原12完整CPU验收。模型/入口和当前执行链fresh复核通过；真实生产M0与正式GPU尚未运行，不将旧CPU玩具检查当M0。

2026-09-29 13:30实际四卡训练；新controller2779171启动2026-09-29T13:27:25.681911+08:00，三个static_none真实M0全119/119、baseline不变、reload0；原100 PID2699844保持27/50。其余12端排队，原12／新15终态验收门保持，无中途best采纳。

2026-09-29 14:00：static_none201与MSVR完整50/best/三路完整图库CPU验收，原MSVR接受行未改；context_none201自己的生产M0通过且初始state与静态控制相同。剩余13端继续，不据无辅助控制改辅助ID权重或取消负端。
