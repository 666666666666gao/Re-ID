# 实验执行表

| Run ID | 用途 | 模型 | 数据集 | 预算 | 指标 | 状态 |
|---|---|---|---|---|---|---|
| H0 | 初值/梯度所有权 | CPU合成direct/vehicle | 合成 | 一次 | 初值、梯度、BN、重载 | NOT_RUN |
| H201 | fused头独立学习 | 原semantic+独立头 | RGBNT201 | prepare/pair/8M0/fresh50/首strict | mAP/R1/R5/R10 | NOT_RUN |
| H310 | fused头独立学习 | 原semantic+独立头 | MSVR310 | 同上 | mAP/R1，附CMC | NOT_RUN |
| H100 | fused头独立学习 | 原semantic+独立头 | RGBNT100 | 同上 | mAP/R1，附CMC | NOT_RUN |
| HR | 一次完整配对报告 | 新三端与旧semantic/global | 全部 | 六全量配对 | query/身份/AP/首位变化 | NOT_RUN |

旧RAW九控制只作为封存输入使用，不重训。该表为启动前登记，实际进度以后以远端campaign与独立进程证据为准。
