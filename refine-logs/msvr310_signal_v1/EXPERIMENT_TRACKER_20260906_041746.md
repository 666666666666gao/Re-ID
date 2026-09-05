# MSVR310 Signal v1 tracker

更新时间：2026-09-06T04:17:46.871857+08:00。PREPARED_PENDING_REMOTE_T0_M0；当前模型训练/检索仍0。

| Run | 目的 | 固定范围 | 状态 |
|---|---|---|---|
| T0 | scene排序回归 | 两个人工可解fixture；远端NumPy，非真实指标 | NOT_RUN |
| M0 | source工程入口 | 三折各8次B64/K8更新，strict reload/48条clean source前向 | NOT_RUN |
| B0 | Signal内部基线 | 三折各50epoch，600query/1032完整gallery，最终一次评价 | NOT_RUN |
| audit | 结果来源与数学 | 完整B0终态后独立审计 | NOT_RUN |
| TriFusion/官方/消融 | 不在本合同范围 | 需独立新合同且保留现有门槛 | NOT_RUN |

详见EXPERIMENT_PLAN.md；不能把数据安装、源码准备、T0或M0称作已测车辆性能。
