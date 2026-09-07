# V28 R2 执行追踪

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V28R2-T0 | T0 | 合成与真实fixture精度回归 | AMP旧/FP32新 | 无新图像 | dt2048导数 | MUST | NOT_RUN | 0更新 |
| V28R2-M0 | M0 | 原工程门 | 两端 | 来源三折 | 203/219梯度等 | MUST | NOT_RUN | 新116步，原FAIL保留 |
| V28R2-Q1 | Q1 | 固定结构配对 | control/joint_tokens | 三折完整图库 | 原五科学门 | MUST | NOT_RUN | M0通过才执行3360步 |

原V28 M0_FAIL不重写；3次诊断重建更新单列。
