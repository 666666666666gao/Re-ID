# V19 执行记录

| Run ID | 阶段 | 目的 | 对比 | 范围 | 指标 | 状态 |
|---|---|---|---|---|---|---|
| V19-T0 | T0 | 新wrapper/optimizer行为 | 冻结/可训练私有尾部 | 远端单元测试 | 输出/重载/storage/更新范围 | TODO |
| V19-M0 | M0 | 真实工程合同 | 三折配对、两端容量、实验端过拟合 | source-only | 梯度/显存/溢出/过拟合/冻结 | TODO |
| V19-Q1 | Q1 | 完整身份隔离主比较 | 三折×两端×20epoch | 3126gallery/571query | 五路mAP/R1/R5/R10/全部AP | CONDITIONAL_M0 |
| V19-AUDIT | 审计 | 全部结果与来源核验 | 全部端 | 完整Q1 | 完整性/科学门 | CONDITIONAL_Q1 |
| V19-D1 | D1 | 同协议dev主结果 | 当前最佳与exact Signal | 141-fit/30-dev | >=65mAP及固定严格胜出 | CONDITIONAL_Q1_PASS |

当前无V19训练或检索结果。只允许seed42、远端GPU；无消融/扫描/official。
冻结方案为同目录EXPERIMENT_PLAN.md；既有V17/V18固定方案与记录保持原样。
