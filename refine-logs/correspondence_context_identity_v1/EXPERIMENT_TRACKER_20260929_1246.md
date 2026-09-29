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
