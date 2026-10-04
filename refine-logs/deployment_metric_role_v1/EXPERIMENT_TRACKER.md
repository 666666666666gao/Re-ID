# 当前执行登记

更新：2026-10-05 §41.842。原队列04:07:25退出1，正式3/6；M0通过3、失败1。source339与控制187不变。原campaign终态保留。

| 数据集 | 条件 | 实际阶段 | 状态/结果 |
|---|---|---|---|
| RGBNT201 | semantic | 8M0/fresh50/2649/首次strict闭合 | E8 74.9363mAP/78.7081R1，配对推进FAIL |
| RGBNT201 | native | 8M0/fresh50/2649/首次strict闭合 | E8 74.9846mAP/77.8708R1，配对推进FAIL |
| MSVR310 | semantic | 8M0/fresh50/706/首次strict闭合 | E38 50.5121mAP/67.8511R1，配对推进FAIL |
| MSVR310 | native | prepare成功；真实8步M0失败 | 缺细节stem首层weight/bias非零梯度，full未启动，无正式分数 |
| RGBNT100 | semantic | 原队列从未prepare；新有限队列登记 | 尚未启动 |
| RGBNT100 | native | 原队列从未prepare；新有限队列登记 | 尚未启动 |

旧86112终态观察器退出0。原supervisor不重启/不回改FAILED；只未启动RGBNT100配对另以相同科学合同继续，PENDING_RGBNT100_QUEUE.py/PENDING_RGBNT100_PLAN.md已登记。此处仍无新实际启动证明。

原六端15对CPU报告0次，不因新两端2/2误写整体6/6。正式best与当前依赖保留；历史不再使用的探针/同组非优seed20权重按回执与SHA退役，原文本/距离/失败记录保留。

仅26GPU0/1，无功率/温度操作；2025原I/O pending不探测。单seed42/已消费官方基准；Goal ACTIVE / UNMET。
