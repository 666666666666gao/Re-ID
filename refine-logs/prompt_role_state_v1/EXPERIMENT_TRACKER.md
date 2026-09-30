# 视觉主干内角色提示状态实验跟踪

当前快照：2026-09-30约01:03，北京时间。六端队列已于00:58:29启动，控制器PID3314702；首批四端真实八批M0均COMPLETE/exit0，随后完整训练运行。两端等待空卡，尚无正式终态。后续按完整50轮、单fused mAP-best、严格重载、完整图库验收；目标ACTIVE／UNMET。

| 数据集 | reset_roles | carry_roles |
|---|---|---|
| RGBNT201 | M0通过；训练7/50 | M0通过；训练6/50 |
| RGBNT100 | M0通过；训练2/50 | PENDING |
| MSVR310 | M0通过；训练9/50 | PENDING |

四卡均在运行首批四端，瞬时利用率74／100／100／100%。此表只是里程碑快照；轮数不等于正式best，也不据中间mAP决定取消任何端。旧跨深度九端负结果仍单独封存。

## 2026-09-30 09:45 完整终态

原六端队列已 `COMPLETE`；六个 child 的 M0／训练50轮／正式评价均 exit0，项目 collector 已将六端标为 `VERIFIED_COMPLETE`。`accepted_matrix.json` SHA256 为 `1405d73124e7a1776de521edefb14620457f7b6e0691787bb18e2f2a8e0af7f9`。三份已保存距离的配对 CPU 诊断也退出0。fresh 独立审计尚在进行；本节不替代其结论。

| 数据集 | reset best／mAP／R1 | carry best／mAP／R1 | carry − reset mAP／R1 |
|---|---|---|---|
| RGBNT201 | 18／71.2650／72.0096 | 30／70.6509／72.3684 | −0.6141／+0.3589 |
| RGBNT100 | 1／85.6486／95.8601 | 1／85.7429／96.3265 | +0.0943／+0.4665 |
| MSVR310 | 15／53.0499／69.0355 | 24／54.2272／70.2200 | +1.1773／+1.1844 |

预登记跨集门槛**未通过**：RGBNT201 mAP 为负。完整四项／两项指标、机制边界与证据路径见 `results/PROMPT_ROLE_STATE_FULL_2026-09-30.md`。不启动仅为 carry 准备的独立global-only阶段。Goal仍为 ACTIVE／UNMET。

2026-09-30 10:10 CST 独立完整审计封存：`REVIEW_FULL_PANEL_20260930.md/json`，原报告 SHA256 `6b067198ec7527c2733b44d5cfda9b834d1d3fdb7b476e59a6cdf75ac1a68644`；裁决 WARN / same-family / provisional，未发现完整性 FAIL。审计不调用项目 scorer，独立复算六端18张矩阵和72个指标、核对全部50轮/20,416步/48步M0/207份源码SHA，并在CPU重建真实模型，六份初始SHA匹配、六份M0 probe和六份best checkpoint严格重载成功，冻结基线未变。没有重跑神经前向。预登记跨集晋级门槛 FAIL（201 mAP carry-reset −0.614069pp），因此不启动 carry-global-only；六端正式结果与审计报告保留。四卡训练队列已结束，不把空闲GPU用来绕过该门槛的追加种子或调参。