# V19 执行记录

| Run ID | 阶段 | 目的 | 对比 | 范围 | 指标 | 状态 |
|---|---|---|---|---|---|---|
| V19-T0 | T0 | 新wrapper/optimizer行为 | 冻结/可训练私有尾部 | 远端单元测试 | 输出/重载/storage/更新范围 | DONE_PASS |
| V19-M0 | M0 | 真实工程合同 | 三折配对、两端容量、实验端过拟合 | source-only | 梯度/显存/溢出/过拟合/冻结 | DONE_PASS |
| V19-Q1 | Q1 | 完整身份隔离主比较 | 三折×两端×20epoch | 3126gallery/571query | 五路mAP/R1/R5/R10/全部AP | RUNNING |
| V19-AUDIT | 审计 | 全部结果与来源核验 | 全部端 | 完整Q1 | 完整性/科学门 | CONDITIONAL_Q1 |
| V19-D1 | D1 | 同协议dev主结果 | 当前最佳与exact Signal | 141-fit/30-dev | >=65mAP及固定严格胜出 | CONDITIONAL_Q1_PASS |

2026-09-05 12:26:41 CST：远端T0四项测试通过，6.95秒；M0全部检查通过。
M0两端203/311训练张量均有有限非零梯度，峰值reserved6478/6814MiB，
overflow0；固定100步超额损失比0.0595140216，冻结state不变。
原始M0快照：`evidence/trifusion_v19_m0_seed42_4b749cd.json`；远端SHA核对
见`evidence/trifusion_v19_m0_transfer_receipt_20260905.json`。
源提交4b749cd，12:15:41在screen `18809.v19_private_tail_4b749cd`启动；
Q1当前fold0对照第9/20epoch，尚无完整最终检索结果。首端约35秒/epoch，
六端连同评价预计75–90分钟，按真实后续时长修正，不因中间值改变预算。
只允许seed42、远端GPU；无消融/扫描/official。D1/dev/official访问仍为0。
冻结方案为同目录EXPERIMENT_PLAN.md；既有V17/V18固定方案与记录保持原样。
