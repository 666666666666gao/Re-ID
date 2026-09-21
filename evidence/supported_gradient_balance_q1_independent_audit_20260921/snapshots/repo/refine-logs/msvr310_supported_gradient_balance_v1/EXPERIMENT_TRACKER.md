# 支持感知梯度平衡工作记录

2026-09-21。R1_ENGINEERING_STOP / R2_M0_CPU_PASS / R2_Q1_FAIL / FULL_AUDIT_IN_PROGRESS，seed42。

| 工作 | 状态 | 证据 |
|---|---|---|
| R1 T0 | PASS | 780来源batch及数学检查 |
| R1 M0 | STOPPED | 3更新；第4步辅助相减误差0.006824872>0.005，05:36:03退出1 |
| R1 Q1 | NOT_RUN | 工程阶段停止，没有检索 |
| R2实现与复审 | PASS_WITH_LIMITS | 直接辅助求导，原控制更新保持；同族原reviewer继续审查，0阻断 |
| R2 T0 | PASS | 05:52:39退出0，780来源batch及数学检查 |
| R2 M0 | PASS_ENGINEERING_ONLY | 06:09:36退出0；248更新、90实际独立参考检查通过 |
| R2 M0_CPU | PASS | 06:09:45退出0；248步/4945920距离元素，全量支持与系数重算 |
| R2 M0独立审查 | WARN/CLOSED_WITH_LIMITS | 独立CPU全量复算通过，0工程/审计阻断；same-family/provisional；实际M0无支持更新0 |
| R2 Q1 | Q1_FAIL | 六端/1560步齐全；fused53.399384→53.452649，+0.053266；配对1/5，Signal1/5 |
| R2 Q1_CPU/Audit | CPU_PASS / AUDIT_IN_PROGRESS | 10:51事后算术修正全量核验通过，原失败保留；新鲜Q1独立审查进行中 |

执行1381639、配置9ce36299，run /root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639，wrapper42758。
完整历史梯度、原网络与cross-scene AP定义不变。仅seed42，先主结果后消融，官方成绩不用于调参。
完整M0文本和汇总见evidence/supported_gradient_balance_m0_complete_20260921及supported_gradient_balance_m0_analysis_20260921。
无新增正式结果，Goal ACTIVE/UNMET。
