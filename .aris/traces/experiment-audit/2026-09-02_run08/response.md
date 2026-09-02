结论：V15 Q1 是有效科学失败，不是完整性失败。

- Overall：WARN（仅因审计时本地 worktree dirty / artifacts 未归档为 clean
  package）
- integrity_status：`valid_q1_fail_with_packaging_warn`
- scientific verdict：Q1 gate failed，V15 sealed
- D1 authorization：false
- claim boundary：只能声称 Q1 按 dev0/official0 的 train-only identity-OOF
  mechanism gate 完成且失败；不能声称 65 mAP、部署收益、official、SOTA、D1
  或消融结果。

## A. GT provenance：PASS

数据记录来自 frozen 141-fit protocol 与 RGBNT201 train 文件。identity/camera
来自数据集文件名与路径，不是模型输出。Q1 heldout 只从 cross-camera eligible
records 筛选；mAP/AP 使用 query/gallery identity 和 camera 计算。

## B. Normalization：PASS

只有标准 embedding L2 normalize + `torch.cdist`。mAP/Rank-1 是 AP/布尔命中
均值乘100。未见按模型自身 max/min/mean 归一化；matched regret 使用 fixed
off comparator 且 stop-gradient。

## C. JSON / console / hash / dirty-state：WARN

JSON 记录 `status=PASS`、`gate.passed=false`、
`next_phase_authorized=false`、`d1_executed=false`；console 最终 JSON 与落盘
JSON 结构化一致。receipt 绑定 commit `71152d3848c05177da0af30b0b921c6a3aa9942a`
和 clean runtime diff；runner/config/M0 SHA 一致。WARN 仅因审计时 checkout
dirty 且 Q1 evidence 未跟踪，不推翻运行时 clean receipt。

## D. Executed/dead path：PASS

Q1 逐 fold 训练、保存 final checkpoint、评价 exchange-on/off，并在返回前计算
gate。`status=PASS` 是运行完成语义，next phase 只取决于 gate。启动失败日志
在 import 前终止，不构成实验结果。

## E. Scope / claim boundary：PASS

冻结计划规定 Q1 是三折 complete-path identity-OOF、21 identities/571 queries、
dev0/official0；Q1 失败必须封存 V15，不运行 D1。Q1 不支持65、部署、official
或 SOTA。

## F. Evaluation type：PASS

类型为 `real_gt_train_only_identity_oof_mechanism`，不是 official test、dev
selection 或 synthetic proxy。dev/official access 均为0。

## Decisive finding

Q1 gate 与冻结计划一致，不是 gate implementation 问题：per-fold fused gains
`[+0.09518,-0.83106,+0.16052]`；weighted fused gain `-0.172068`；bootstrap
95% LB `-0.950330`；CNN/Transformer aggregate negative，Mamba positive。
fused strictly beats branches 为 true，但不足以通过全门。

因此 `status=PASS` 只表示 runner 完成；`gate.passed=false` 是科学判定；
`next_phase_authorized=false`；D1 不授权、不应执行。
