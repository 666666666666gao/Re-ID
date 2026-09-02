# 研究方案：V14 Fold-Robust Retrieval-Regret Router

## Problem Anchor

- **底线问题**：在远端单张 RTX 3090 上，基于 RGBNT201 固定
  `141-fit / 30-dev`、seed 42 和 exact Signal 3072D 前缀，让 CNN、
  Transformer、Mamba 三类预训练残差专家形成可部署的动态融合，而不是
  继续被近静态 Router 平均。
- **必须解决的瓶颈**：V8 已证明专家具有 query 级互补性；V13 的九路
  点式 utility-KL 目标却近乎均匀且随 fold 漂移，不能稳定把互补性转成
  held-out 检索增益。
- **非目标**：本阶段不增加新 backbone、不启用 HFER、不复现 baseline、
  不做多种子、不做消融、不访问 official test，也不扫描温度、epoch、LR、
  margin、fold 或损失权重。
- **成功条件**：先通过严格 train-only OOF Router 门；只有通过后才允许
  一次 dev。dev fused 必须 mAP≥65，且 mAP/Rank-1 严格超过 exact Signal、
  当前可部署最好结果和 CNN/Transformer/Mamba 三个分支。

## 证据锚点

V13 target-learnability 诊断使用 571 个 fit-only query、21 个身份和三折，
训练为 false、optimizer step 为 0、dev/official access 均为 0。九路 target
平均归一化熵 `0.99983197`，平均最大概率 `0.11527554`（均匀值
`0.11111111`），Top1-Top2 utility 中位差 `0.00034070`，fold 槽位排序相关
`-0.50 / 0.40 / 0.05`。这些证据只支持删除点式 utility-KL，不支持预言
V14 能达到 65 mAP。

## 方法假设与冻结边界

> 当专家、部署输入、融合能量和质量控制全部固定时，在每个 OOF teacher
> 坐标系内直接优化跨相机检索 regret，并由最坏 source fold 控制更新，会比
> V13 的近均匀点式动作蒸馏更稳定地迁移到 held-out fit fold。

保持共享三模态几何、exact frozen Signal、V8 Phase-A CNN/Transformer/
Mamba residual experts、matched-token residual、层级 Router、missing mass=0、
fixed `alpha=0.2`、V13 optimizer/100 epochs 和质量损失不变。Router 输入是
冻结的 all-fit deployment features；只有 teacher/replay embedding 是
identity-OOF。本实验不是完整路径 identity-OOF 泛化实验。HFER 保持关闭。

## Fold-bound 检索风险

所有损失使用：

```text
risk(fold_id, rows_in_fold_id, features_from_generator_fold_id, weights)
```

row、fold、OOF generator 必须一致，query/gallery 只来自同一 fold，禁止跨
generator 距离。embedding 经 exact V13 fusion 和 L2 normalize 后：

```text
d_pos(i) = max_j ||z_i-z_j||_2, y_j=y_i 且 camera_j!=camera_i
d_neg(i) = min_k ||z_i-z_k||_2, y_k!=y_i
R_f(w)   = mean_i softplus(d_pos(i)-d_neg(i)).
```

不增加 margin、temperature、listwise relaxation 或新损失权重。

## Source-only comparator 与训练目标

留出 fold `h`，source folds 为 `a,b`：

```text
s* = argmin_s max(R_a(one_hot(s)), R_b(one_hot(s))), s∈{0,...,8}
G_f(w) = R_f(w) - stop_gradient(R_f(one_hot(s*)))
L_total = max(G_a(w), G_b(w)) + L_quality.
```

`s*` 只由 source folds 选择。九槽枚举是 comparator 定义，不是调参。V13
utility tensor、softmax temperature 与 KL 不参与优化，只作诊断。

## Q1 门禁

在 held-out fold `h` 上定义：

```text
risk_gain_h   = R_h(one_hot(s*)) - R_h(learned)
ap_gain_h     = AP_h(learned) - AP_h(fixed)
margin_gain_h = margin_h(learned) - margin_h(fixed).
```

必须满足：每折 risk gain>0；每折 AP/margin gain≥0；三项 pooled paired
identity-cluster bootstrap 95% 下界均>0；三模态退化后质量下降；missing mass
严格为0；Phase-A SHA 不变；dev0、official0。Replay Rank-1、V13 utility/
action Top-1 和 held-out 自身最佳 fixed slot 仅报告、不门控、不参与选择。

## 执行与终止

1. **M0**：手算 risk、跨相机正样本、fold binding、梯度覆盖、exact fusion
   和范围回执测试。
2. **Q0**：exact cache 零优化步，验证三折风险支持、有限非零梯度、SHA、
   dev0/official0。
3. **Q1**：唯一 seed42 三折 OOF qualification；任一门失败即封存，不扫描。
4. **条件 final refit/dev**：Q1 全通过后才以三折各自 OOF risk 的最大 regret
   refit 一次并做一次 dev；feature vector 和距离永不跨 fold 混合。

dev fused 必须 mAP≥65，且 mAP/Rank-1 严格超过 Signal、V8 Phase-B 和三
分支。此前禁止 official test、消融、多种子和 baseline rerun。

Q1 通过最多支持 all-fit deployment input + identity-OOF teacher/replay 范围
下的三 fit-fold 转移；不支持完整路径 OOF、HFER、dev、SOTA 或因果主张。
