修正版 V15 M0 复审结论：Overall `WARN`，integrity_status
`warn_packaging_only`。科学/协议完整性通过；唯一 WARN 是当前本地工作树仍有
未提交/未跟踪文件。`Q1_authorized: true`，但授权边界是：用这份 hash-bound
M0 receipt，在 clean `1f2de44f0c7c953bea7d75921be509ce9704f84c` 状态启动
Q1；不授权 D1、dev 性能声明、official、multi-seed、ablation 或扫描。

## A. Ground-truth provenance: PASS

M0 从 RGBNT201 配置和 train records 构造。训练标签和 camera IDs 来自
`_training_batch` 与 batch。loss 里 labels/cameras 作为监督和 batch retrieval
risk 输入。未发现把模型输出当 GT。

## B. Score/loss normalization: PASS

无自归一化指标欺诈。retrieval risk 使用 L2 normalize + cdist + identity/camera
mask。matched regret 对 off comparator stop-grad。修复新增的 floor 是固定 off
comparator 下界，不是用预测最大值/均值做分母。

## C. Result/log/hash/dirty-state: WARN

Result 与 console 内嵌 JSON 逐字一致，`objects_equal=true`。receipt 为
`status: PASS`、`passed:true`，绑定 clean runtime diff、runner SHA 和 config
SHA。WARN：审计时本地工作树有尚待提交的 evidence/results/tracker 文件。这个
不推翻已记录的 M0 receipt，但 Q1 启动前应确保运行状态被 clean commit 或明确
diff 绑定。

## D. Dead-code / executed M0 gates: PASS

修复后的 M0 gate 是显式函数。容量门只要求 no overflow、frozen unchanged、
`live_exchange_stages == (0,1)`，不再要求 8-step `110/110`。100-step overfit
仍要求所有 trainable tensors reached 和 ratio 过门。

## E. Scope/claim boundary: PASS

配置仍是 B64/K8、8-step capacity、100-step overfit、Q1/D1 final-only，且
dev/official/rerank 禁止。receipt dev0/official0。tracker 写明 M0 PASS、Q1
authorized、D1 blocked by Q1，并禁止 official/multi-seed/scans。

## F. Evaluation type: PASS

这是 `real_gt train-only engineering qualification`，不是 dev performance
eval。容量/overfit 用 train records；dev/official access 都为 0。Q1 仍只是
mechanism qualification，D1 才是唯一 deployable/dev claim source。

## Decisive finding: PASS_WITH_PACKAGING_WARN

修复严格对应上次两项 `INVALID_GATE_IMPLEMENTATION`：

1. 旧的 capacity 必须无 missing tensor 被替换为两层 exchange live。receipt
   保留 `107/110` 和 3 个 missing tensors，但 `live_exchange_stages=[0,1]`；
   100-step 为 `110/110`。
2. 修复后用
   \(F=F_{CE}+\frac14\sum_o\mathrm{softplus}(-R_o^{off})\) 作为 conservative
   lower bound。receipt 的 label floor `0.57838292104621`，matched-regret floor
   `0.4744258522987366`，combined floor `1.0528087733449465`，复算 ratio
   `0.0515540985`，低于门槛 `0.1`。

无 dev/official 泄漏；floor 在 fixed train batch 上用 off embeddings、labels、
camera IDs 计算。

**Q1 authorization: YES.** M0 engineering gate 现在有效通过，tracker 一致，
Q1 runner 只要求 passing M0 receipt。授权不延伸到 D1 或任何性能/SOTA 声明。
