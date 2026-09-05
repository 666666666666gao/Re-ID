# V24 experiment tracker

更新时间：2026-09-06T01:29:21.845569+08:00。状态 M0_PASS_Q1_RUNNING；独立 M0 审计待完成。
实际进度观察时间：2026-09-06T01:10:42+08:00；未将当前墙钟时间冒充新的 GPU 进度。

| 项目 | 状态 | 证据 |
|---|---|---|
| source 成员合同 / 代码配置计划登记 | DONE_BOUND_6a4ac2c | evidence/trifusion_v24_preregistration_20260906.json |
| CUDA T0 六机制用例 | DONE_PASS_SYNTHETIC_ONLY | evidence/trifusion_v24_t0_20260906.json/log/xml |
| 三折六端预检 | DONE_PASS_ENGINEERING_ONLY | evidence/trifusion_v24_preflight_metadata_verification_20260906.json |
| 完整 M0 容量及固定 100 步 | DONE_PASS_ENGINEERING_ONLY | results/TRIFUSION_RGBNT201_V24_COMPLETE_M0_2026-09-06.md |
| 全 M0 数组及文件核验 | DONE_116_ROWS_47_FILES | evidence/trifusion_v24_m0_array_verification_20260906.json; evidence/trifusion_v24_m0_file_verification_20260906.json |
| 独立 M0 审计 | PENDING | EXPERIMENT_AUDIT_V24_M0.md/json 尚未完成 |
| 六端 Q1 | RUNNING_ORIGINAL_PID52030 | evidence/trifusion_v24_progress_20260906_011042.json |
| D1 / dev / official / 消融 | NOT_QUALIFIED_NOT_RUN | 等原定完整 Q1 科学门 |

执行 commit 6a4ac2cd95af2ca1a9122d1f79aabd3a83e4fe33；screen v24_source_prototype_6a4ac2c。
00:56:14 启动；完整 M0 共 18882 次 source 记录前向、153 次 source batch 前向、96 次双视图预检前向、
116 次更新 / 232 次视图前向反向。固定第 100 步 excess 比 0.059999361786045424；五项工程门全部通过。
参数均 98,800,141 总量 / 7,841,292 可训练 / 203 张量；新增推理参数 0。
原型 108 × 7680，包含每折全部 94 个 source 身份，分阶段/折/端点重置；所有真实跨相机同 ID 保持正关系。
01:10 观察只有 Q1 首端 1 epoch /29 更新；未完成配对 fold，尚无 Q1 科学结果。
原计划六端全部训练、终态比较及失败封存合同保持；V23 Q1_FAIL 既有封存不变。
