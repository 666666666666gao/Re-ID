# Context/local identity执行跟踪

2026-09-29：固定seed42，三个数据集各五条件；全部full50/官方mAP-best/独立重载。

| 条件 | RGBNT201 | RGBNT100 | MSVR310 |
|---|---|---|---|
| static_none | READY_NOT_RUN | READY_NOT_RUN | READY_NOT_RUN |
| context_none | READY_NOT_RUN | READY_NOT_RUN | READY_NOT_RUN |
| static_local | READY_NOT_RUN | READY_NOT_RUN | READY_NOT_RUN |
| context_local | READY_NOT_RUN | READY_NOT_RUN | READY_NOT_RUN |
| context_global | READY_NOT_RUN | READY_NOT_RUN | READY_NOT_RUN |

生产M0、formal队列均未启动。READY_NOT_RUN不等于M0通过或正式成绩。

2026-09-29 13:10：五份独立源码实现；CPU合成初始化/查询/八步梯度检查通过，所有五条件原主输出一致；真实M0未运行，15端未启动。前序固定为原M3全12端完成且CPU验收通过，不抢占或提前启动新GPU任务。执行链fresh复核进行中。

2026-09-29 13:20启动前登记：原M3已验收11/12，只剩100 matched/predictor训练。为兑现四卡利用率，新15在空卡接续；旧worker GPU在train→evaluate阶段切换也保留，worker前等待其完成。新15终态前仍要求原12完整CPU验收。模型/入口和当前执行链fresh复核通过；真实生产M0与正式GPU尚未运行，不将旧CPU玩具检查当M0。
