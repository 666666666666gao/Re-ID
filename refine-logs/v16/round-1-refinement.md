# V16 Round 1 Refinement

## Problem Anchor

- **Bottom-line problem**：在固定 RGBNT201 `141-fit / 30-dev` 协议上，让共享
  CLIP 强语义主干上的 CNN、Transformer、Mamba 三个异构专家形成可泛化的真实
  互促，使部署 fused 严格高于 exact Signal、当前 V8 deployable comparator 和
  三个分支，并达到至少 `65 mAP`；只有同协议实验确实超过公开最佳时才宣称
  SOTA。
- **Must-solve bottleneck**：V8 已证明三专家有显著 query-level Oracle 互补，
  但 V8–V14 的 sample-local Router 不能稳定预测跨样本检索效用；V15 又证明
  inference-time hidden-vector exchange 的 10/12 条 stage-edge 在身份折间发生
  极性反转，并使 fused 改善/受损 query 为 `87/141`。必须把互补知识转成
  identity-disjoint 可迁移的检索关系，而不能再次依赖 Router、静态特征注入或
  更大的末端融合器。
- **Non-goals**：不调 V15 edge scale、LR、epoch、regret 或 checkpoint；不重跑
  Signal baseline；不增加 Router、HFER/CRDE、DINO、文本、额外 backbone、
  reranking 或 test-time training；主结果通过前不做消融、多 seed 或 official
  test；不把 fit-only Oracle 当成部署结果。
- **Constraints**：全部训练与评估仅在远端单张 RTX 3090 24GB；seed42；共享
  三模态几何增强；真实 B64/K8；exact Signal 3072D 前缀逐元素保留；复用 V12
  complete-path identity-OOF checkpoints 和 V8 all-fit Phase-A checkpoint；先过
  train-only identity-disjoint Q1，之后才允许唯一 D1。
- **Success condition**：V16 在三个完整路径 OOF fold 上相对同 checkpoint、同
  训练预算、无 SATR 的 comparator 得到稳定的 fused 与分支增益，聚合 fused
  mAP 增益至少 `+1.0` 且 identity-cluster bootstrap 95% 下界大于 0；随后唯一
  seed42 D1 的 fused `>=65 mAP`，并严格超过 exact Signal、V8 deployable
  comparator、CNN、Transformer、Mamba。

## Anchor Check

- **Original bottleneck**：Router 无法从 sample-local input 预测跨样本效用，
  hidden-vector exchange 的方向又在 unseen identity folds 上反转。
- **Why this still solves it**：只转移在同一个 Signal-hard cross-camera relation
  上由另外两支共同验证的排序余量；不转坐标向量，也不学习部署路由。
- **Rejected drift**：没有接受任何 Router、EMA teacher、额外 backbone、DINO、
  文本或多实验菜单建议。

## Simplicity Check

- **Dominant contribution**：只有 SATR 一项——Signal anchor、two-peer
  intersection、one-sided repair 是一个不可拆的布尔选择与损失。
- **Removed/merged**：把 `L_protect` 降为 fused safety constraint；把 B³ 全
  triplet 枚举收敛为每个 query 一个 exact-Signal batch-hard relation；没有新
  trainable module。
- **Necessary extra cost**：接受 matched no-SATR endpoint。它不是扩张实验菜单，
  而是把“继续训练”与“SATR”分开的最小因果对照。

## Changes Made

1. **公平 comparator**：每 fold 从逐 tensor 相同初态分别训练 SATR 和
   no-SATR；重置 RNG，保持 sampler/augmentation/optimizer/schedule 完全一致，
   唯一差异为 relation loss 权重。
2. **单个 Signal-hard relation**：用 exact Signal 为每个 query 选最难异 camera
   positive 和最近 negative，所有专家在同一个 `(q,p0,n0)` 上比较；coverage
   分母变成有效 query 数。
3. **固定常数**：基于三折、每折八个 B64/K8 train-only batch 的 optimizer-0
   probe，冻结 `delta_r=0.05`、`gamma_p=0.30`、`epsilon_p=0.02`、
   `lambda_r=1.0`、`lambda_p=0.25`，不做扫描。
4. **可执行 gate**：每 fold/receiver 聚合训练 coverage 必须在
   `[0.5%,25%]`；否则 fail-closed，无 single-peer fallback。

## Revised Proposal

### Technical Gap

V7/V8 已完成用户对 V5/V6 提出的共享几何、matched residual、层级
`P(m|x)P(e|m,x)`、有界 alpha、normalized Triplet、label smoothing 和 B64/K8。
V8 branch Oracle 达 `64.7850 mAP`，但 V8–V14 learned Router 均未把 Oracle
空间可靠转成部署增益；V15 fused/CNN/Transformer/Mamba 的 matched Q1 增益为
`-0.1721/-0.1576/-0.2606/+0.2898`，且 10/12 stage-edge 的符号跨 fold 不稳定。

Similarity KD、RKD、PMKD 和旧 RDPT 已覆盖关系蒸馏、困难样本与选择式教学；
所以 V16 的贡献边界只允许是：**用 exact foundation baseline 定义同一检索
relation，由两个异构 peer 的交集验证改进，并对 receiver 做单侧排序修复。**

### Method Thesis

对每个 query，只修复 exact Signal 当前最难的一条跨相机正负关系；只有另外两
支都在同一关系上严格超过 Signal 与 receiver，receiver 才接收两支共同 margin
下界。训练结束后删除全部协同计算，三个 private embeddings 以 fixed bank
推理。

### Complexity Budget

- 冻结/reuse：Signal、CLIP tail、V8/V12 expert topology、共享几何与 evaluator；
- 开放：V8 已有 CNN/T/M role adapters、residual projections、训练 necks/
  classifiers；Signal、shared CLIP tail 和 exact prefix 始终冻结；
- 新参数：0；新推理 FLOPs：0；
- 禁止：Router、reliability、HFER/CRDE、EMA、teacher checkpoint、projection
  adapter、learned fusion/alpha、logit KL、embedding matching。

### System Overview

```text
RGB/NIR/TIR -> shared geometry -> frozen Signal -> exact z0
                         |
                  shared CLIP block8
             / CNN       | Transformer      \ Mamba
            z_C           z_T                 z_M
              \  training-only Signal-hard margins  /
               two-peer intersection -> receiver repair
                              |
        inference: exact z0 + fixed normalized private residual bank
```

### Exact Signal-hard Relation

对 L2-normalized exact Signal embedding `z0`，batch 内每个 query `q` 定义：

\[
p_0(q)=\arg\min_{p:y_p=y_q,c_p\ne c_q}s(z_0^q,z_0^p),\qquad
n_0(q)=\arg\max_{n:y_n\ne y_q}s(z_0^q,z_0^n).
\]

若不存在异 camera positive，该 query 从 SATR/protect 中排除，不创建替代 positive。
所有专家在同一对 `(p0,n0)` 上计算：

\[
m_e(q)=s(z_e^q,z_e^{p_0})-s(z_e^q,z_e^{n_0}),
\quad e\in\{0,C,T,M,fused\}.
\]

这避免每支选择不同 hard pair 后的不可比性，也把 relation budget 固定为每 batch
至多 B 条。

### SATR 单一布尔合同

对 receiver `e`，另两支为 `j,k`。teacher 和 mask 全部 detached：

\[
\bar t_e=\min(\operatorname{sg}(m_j),\operatorname{sg}(m_k)),
\]

\[
a_e=
[\bar t_e>0]\land
[\bar t_e\ge
\max(\operatorname{sg}(m_0),\operatorname{sg}(m_e))+0.05].
\]

live receiver margin 只出现在 loss：

\[
L_{SATR}^e=
\frac{\sum_q a_e(q)\,
\frac12\max(0,\bar t_e(q)-m_e(q))^2}
{\max(1,\sum_q a_e(q))}.
\]

空 mask 精确返回0；不使用 single-peer fallback。CNN、Transformer、Mamba 都
轮流做 receiver，但每个 query 可以只有一个方向、多个方向或全部拒绝。

### Fused Safety Constraint

对 Signal 已可靠的同一 hard relation：

\[
a_p=[\operatorname{sg}(m_0)\ge0.30],
\]

\[
L_{protect}=
\frac{\sum_q a_p(q)\max(0,
\operatorname{sg}(m_0(q))-0.02-m_{fused}(q))^2}
{\max(1,\sum_q a_p(q))}.
\]

它只是防退化约束，不是第二贡献，也不是 fused gain 的充分条件。是否真正转成
fused mAP 只能由 Q1/D1 证明。

### Frozen Training Objective

复用 V15 已注册 ReID 权重：

- fused ID/triplet：`0.25 / 1.0`；
- 每 branch ID/triplet：`1/12 / 0.25`；
- 每 residual ID/triplet：`1/12 / 0.25`；
- triplet margin `0.3`，label smoothing `0.1`。

新增且只新增：

\[
L=L_{ReID}+1.0\sum_eL_{SATR}^e+0.25L_{protect}.
\]

`delta_r=0.05, gamma_p=0.30, epsilon_p=0.02` 和所有 loss weights 在 Q1
前冻结，不看 dev，不扫描。

### Threshold Evidence and Activity Gate

三折各八个 deterministic source-training B64/K8 batch 的 optimizer-0 probe 显示，
`delta_r=0.05` 下 CNN/T/M coverage 分别为：

- fold0 `1.56/3.13/4.69%`；
- fold1 `3.13/1.56/14.06%`；
- fold2 `4.17/1.39/5.56%`。

`m0>=0.30` 的 protect coverage 为 `48.44/26.56/58.33%`。因此预注册：每
fold/receiver 在完整20 epoch训练轨迹上的 eligible-valid-query 比例必须落在
`[0.5%,25%]`。范围外即 activity gate FAIL，但不修改阈值或启用 fallback。

### Fair Q1

每个 V12 fold 建立两个端点：

1. `SATR`：上述完整 loss；
2. `matched no-SATR`：`lambda_r=lambda_p=0`。

两者必须：

- 从同一完整初始化 state SHA 构建；
- 使用同一94 source identities、B64/K8、20 epochs、AdamW LR `3.5e-4`、
  weight decay `1e-4`、AMP init scale256、相同 trainable tensor 集；
- 在每个端点前重置 seed42，使 sampler、几何增强和 modality-quality draw 序列
  一致；
- final epoch only，不选 checkpoint；
- 在同一47 unseen identities 上 paired evaluation；dev0/official0。

这样 matched gain 唯一归因差异是 SATR/protect loss，而不是继续 fine-tuning。

### Inference

训练期 relation path 完全删除。使用无 Router/无 exchange 的 fixed 表示：

\[
z_{fused}=[z_0,\|z_0\|\operatorname{Normalize}
(\operatorname{Concat}(r_C,r_T,r_M))].
\]

输出 exact baseline_only、fused、CNN、Transformer、Mamba；三个 expert 坐标
从未被彼此匹配。

### Novelty Boundary

SATR 不声称 relational KD、mutual KD、hard sample mining 或 teacher selection
本身新颖。它与 RDPT 的关键差异是：无 reliability、无单 peer、无 KL/role
payload/private hinge；只在 exact Signal 选择的同一 cross-camera relation 上，
使用 two-peer lower-bound intersection 做 one-sided repair。该组合直接由 V15
“hidden direction 跨身份反转”的失败证据导出。

### Claim-Driven Validation

**Claim 1：SATR 产生 identity-disjoint 的三方互促。**

- 三个 Q1 folds，SATR 对 matched no-SATR；
- 每折 fused mAP gain `>=0`；聚合 fused `>=+1.0 mAP`；21-identity bootstrap
  lower bound `>0`；aggregate CNN/T/M gain 全 `>0`；每折至少两支正增益；
- 每 fold/receiver coverage 在 `[0.5%,25%]`；
- fused 严格高于 SATR 三分支。

**Claim 2：训练期协同转为无需动态模块的部署增益。**

- 仅 Q1 全门通过后，从 V8 Phase-A all-fit checkpoint 运行同一 SATR 配方20
  epochs，final-only dev；
- fused `>=65 mAP`，严格高于 exact Signal、V8 Phase-B `58.4050` 和 V16
  CNN/T/M；Rank-1 严格高于三分支；official0。

### Failure Boundaries

- coverage 稀疏/饱和：fail-closed，不调阈值；
- 两 peer 共误：GT-defined relation、`t>0` 和 Signal comparison共同限制；
- branch 改善未转 fused：由 fused Q1/D1 gate 否定，不加 Router/alpha；
- 只帮助一支：aggregate 三 branch gate 否定 mutual-promotion claim；
- Q1 失败：封存 V16，不访问 dev。

### Compute

- M0：15–25分钟；
- Q1：公平双端点使三折训练量翻倍，估计70–110分钟；
- D1（条件执行）：20–30分钟 + 最终dev约1分钟；
- 晋级前约2 GPU-hours，全部在单 RTX3090。

