# V25 experiment tracker

更新时间：2026-09-06T23:00:03.502087+08:00；执行commit97468dd，原PID112550。
最新活跃证据：22:54:32.819478，非根据旧lock或意图推定；尚无退出文件。

| 阶段 | 状态 | 完整范围 |
|---|---|---|
| SOURCE-METADATA / T0 | COMPLETE_PASS | 3360批全重放、全部source身份/记录覆盖 |
| M0 | COMPLETE_PASS | 48只读batch；16容量+100固定batch优化，全203梯度、overflow0 |
| Q1 | RUNNING | 截至22:54:32，19/120epoch，fold0-control19/20；全部六端继续 |
| 终态核验器 | PREPARED_NOT_RUN | 全训练行/全距离/全排名/571query/21ID，源SHAcc1cf0c1db52dc3b9bf770cd9cdc427d9142f751b62afda78bb5a73fa4e5a277 |
| D1 / official / ablations | NOT_QUALIFIED_NOT_RUN | 不改科学门、不进行官方调参或消融 |

过拟合超额loss比0.03988262041085136；原阈值0.1，固定下界0.57838292104621。
两种容量reserved6054/6198MiB；全部阶段冻结state不变。
根据5至19epoch约442秒，约31.6秒/epoch；粗估六端在23:45–次日00:00完成，按实际终点时间修正。
长任务按180–300秒或预计里程碑观察原PID，不因观察超时重启。
最近datafree22.014GiB；已清理24冗余resume权重的结果保持。
