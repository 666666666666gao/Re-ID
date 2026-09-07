# V28 实验追踪

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V28-T0 | T0 | 实际Mamba代数与梯度 | 合成1152Token | 无数据 | 恒等/梯度/参数量 | MUST | PASS | 实测413056参数/16张量 |
| V28-M0 | M0 | 完整接线与固定100步 | control/joint_tokens | 来源三折 | 原工程门 | MUST | M0_FAIL | 116更新，dt_proj.weight梯度始终零 |
| V28-Q1 | Q1 | 单结构完整配对 | 两端各20epoch | 原三折完整图库 | 五输出/五门 | MUST | SKIPPED_M0_FAIL | 0 Q1更新 |
| V28-VERIFY | 终态 | 全距离排序及训练复算 | 六终点与完整日志 | 全571query/21IDs | 完整性及增减 | MUST | NOT_EXECUTED | 没有六端检索数组 |

V28原M0已结束并封存；另行精度诊断不恢复本轮Q1。
