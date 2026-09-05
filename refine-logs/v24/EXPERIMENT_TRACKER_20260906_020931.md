# V24 experiment tracker

更新时间：2026-09-06T02:09:31.389952+08:00。状态 M0_PASS_Q1_RUNNING；独立 M0 审计已完成，整体 WARN、工程 PASS。
最近实际 GPU 观察：2026-09-06T01:53:22.514115+08:00；未将本次文档更新时间当作新的训练进度。

| 项目 | 状态 | 证据 |
|---|---|---|
| source 成员合同 / 代码配置计划登记 | DONE_BOUND_6a4ac2c | evidence/trifusion_v24_preregistration_20260906.json |
| CUDA T0 六机制用例 | DONE_PASS_SYNTHETIC_ONLY | evidence/trifusion_v24_t0_20260906.json/log/xml |
| 三折六端预检 | DONE_PASS_ENGINEERING_ONLY | evidence/trifusion_v24_preflight_metadata_verification_20260906.json |
| 完整 M0 容量及固定 100 步 | DONE_PASS_ENGINEERING_ONLY | results/TRIFUSION_RGBNT201_V24_COMPLETE_M0_2026-09-06.md |
| 全 M0 数组及文件核验 | DONE_116_ROWS_47_FILES | evidence/trifusion_v24_m0_array_verification_20260906.json; evidence/trifusion_v24_m0_file_verification_20260906.json |
| 独立 M0 审计 | DONE_WARN_ENGINEERING_PASS | EXPERIMENT_AUDIT_V24_M0.md/json; evidence/trifusion_v24_m0_audit_closure_20260906.json |
| 六端 Q1 | RUNNING_ONE_PAIRED_FOLD_COMPLETE | refine-logs/v24/PROGRESS_20260906_015322.md |
| 每折 fused 非负必要门 | VIOLATED_BY_FOLD0 | 首折 fused 增益 -0.1251440758834974 pp，仍完成全部六端 |
| 完整 Q1 终态 / 三折聚合 / 独立终态审计 | PENDING | 原进程继续剩余四端；尚无完整终态 |
| D1 / dev / official / 消融 | NOT_QUALIFIED_NOT_RUN | 第一折已违反必要门 |

执行 commit 6a4ac2cd95af2ca1a9122d1f79aabd3a83e4fe33；原 PID52030，screen v24_source_prototype_6a4ac2c。
00:56:14 启动；完整 M0 共 18882 次 source 记录前向、153 次 source batch 前向、96 次双视图预检前向，
116 次更新 /232 次视图前向反向。固定第 100 步 excess 比 0.059999361786045424；五项工程门全部通过。
参数均 98,800,141 总量 /7,841,292 可训练 /203 张量；新增推理参数 0。
原型 108 × 7680，包含每折全部 94 个 source 身份，分阶段/折/端点重置；所有真实跨相机同 ID 保持正关系。

独立 M0 审计依据 01:10 快照：42 项原始输入哈希全匹配，116 行损失复算的最大差异为 1.043081283569336e-7，
固定下界和第 100 步比值完全一致。整体 WARN；工程与固定 M0 资格通过，不构成检索收益证据。
保留远端二进制仅凭完整文件回执、参数未由审计者实例化复算、同家族审计的范围限制。
固定批次只更新 9/108 个原型，不能证明完整训练中的记忆新鲜度。审计原文和输入快照已存 run14。

01:53 观察第一对完成 40 epoch /1160 更新；190 query、1000 完整 gallery。
fused 72.72132545325914 →72.59618137737564；CNN/T/M 增益 +0.09201704604836891/-0.19233788066681257/-0.4895198885976413。
五输出 AP/Rank、40 行 epoch、两端全部 580 步采样及 20 次原型年龄/更新计数已核验。
这是晚于独立 M0 审计快照的第一折结果，不把它写成已独立审计的完整三折结论。
按该阶段实际耗时，下一成对折预计 02:37，完整终态约 03:20–03:35；预测不等于实测完成。
原六端完整合同、五项门、seed42 和失败封存规则不变，不修改 V24 或重启。
