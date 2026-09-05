# MSVR310 Signal v1 tracker

更新时间：2026-09-06T04:17:46.871857+08:00。T0_PASS_M0_LAUNCHED；尚无正式基线或车辆检索结果。

| Run | 目的 | 固定范围 | 状态 |
|---|---|---|---|
| T0 | scene排序回归 | 两个人工可解fixture及1032真实标签掩码；远端NumPy，非真实检索指标 | PASS |
| M0 | source工程入口 | 三折各8次B64/K8更新，strict reload/48条clean source前向 | LAUNCHED_NO_TERMINAL_YET |
| B0 | Signal内部基线 | 三折各50epoch，600query/1032完整gallery，最终一次评价 | NOT_RUN |
| audit | 结果来源与数学 | 完整B0终态后独立审计 | NOT_RUN |
| TriFusion/官方/消融 | 不在本合同范围 | 需独立新合同且保留现有门槛 | NOT_RUN |

详见EXPERIMENT_PLAN.md；不能把数据安装、源码准备、T0或M0称作已测车辆性能。

实际启动观察2026-09-06T04:25:50.148612+08:00，代码2dcbe85，wrapper PID61639；
启动前GPU free24126MiB/used1MiB/util0。尚未读取M0终态，不提前填PASS或实际更新数。
T0两测试PASS，0.611659秒；1032真实训练记录query资格/正例掩码全部核对。
首次计划检查约04:29:50；正式三折50epoch尚未启动。此次更新记录时间2026-09-06T04:28:08.218395+08:00。
