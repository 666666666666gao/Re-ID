# V16 Round 2 Refinement

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

- 原始瓶颈仍是“互补存在，但 Router 不可泛化且 hidden exchange 跨身份反转”。
- SATR 仍只传坐标无关的 relation；没有回到 Router、inference exchange 或额外
  foundation model。
- Round 2 只修正 activity gate 的判定时点，不改变问题或方法。

## Simplicity Check

- 主贡献仍只有一个无参数 SATR mask/loss；protect 只是同一 margin 张量上的
  safety constraint。
- coverage 不再承担训练结果门，只证明 step0 教学信号存在；没有新增 fallback。
- paired draw hash 是公平性 receipt，不是模型模块。

## Changes Made

1. **Activity gate 固定到初态**：`[0.5%,25%]` 只在 deterministic optimizer-0
   pass 判定。20 epoch coverage 仅记录，下降不会否定成功修复。
2. **完整阈值披露**：`delta=0.05` 未试其他值；`gamma=0.10` 在 fold0 保护100%
   后一次性改成 `0.30` 并做三折冻结；其余常数未执行替代值。
3. **Camera 与空 query**：cross-camera positive 使用 RGBNT201 物理 camera ID；
   没有异 camera positive 的 query 排除，negative 只需异身份，无 fallback。
4. **公平 draw receipt**：M0 与 Q1 记录两个端点的 initial state、ordered sample
   path、sampler index/seed 和前八个 post-transform batch tensor SHA；不相等即
   Q1 integrity FAIL。

## Revised Proposal

### Technical Gap

用户对 V5/V6 的主要修复已在 V7/V8 实现，包括共享几何、matched residual、
层级 routing、有界 alpha、normalized Triplet、label smoothing 和真实 B64/K8。
V8 branch Oracle 达 `64.7850 mAP`，证明 CNN/T/M 的 query-level 互补；但
V8–V14 的多种 Router target/loss 均未跨身份泛化。V15 取消 Router 后，静态
role-delta exchange 仍使 fused/CNN/T/M 聚合变化为
`-0.1721/-0.1576/-0.2606/+0.2898`，且10/12条边跨fold反号。

因此最小充分干预是：不传 feature、不选 deployment expert，只在训练期传递
由 exact Signal 与另外两支共同验证的 cross-camera ordering relation。

### One-sentence Thesis

对 exact Signal 在当前 batch 中最难的跨相机正负关系，receiver 只有在另外两
个异构专家都严格胜过 Signal 与 receiver 时，才补足两支共同 margin 下界；
训练结束后完全删除协同路径，以 fixed private bank 推理。

### Contribution Boundary

- **Dominant contribution**：Signal-Anchored Triadic Repair（SATR）——
  Signal-hard relation、two-peer intersection、one-sided live-receiver repair
  构成单一机制。
- **Safety constraint**：保护 Signal 已可靠的 fused relation，不作为创新点。
- **Existing foundation**：V8 pretrained-tail semantic rebranching 与 exact
  Signal prefix 保留，但不冒充 V16 新发明。
- **不声称**：首次 relational/mutual/hard-sample KD、首次异构三分支或首次
  teacher selection。

### Complexity Budget

- Frozen：Signal、exact 3072D prefix、shared CLIP tail；
- Same-scope trainable in both Q1 endpoints：既有 CNN/T/M role adapters、
  residual projections、BN/classifier heads；
- New trainable parameters / inference modules：`0 / 0`；
- Explicitly absent：Router、reliability、HFER/CRDE、EMA、extra teacher、
  learned alpha/fusion、logit KL、embedding/token matching、DINO/text。

### Shared Signal-hard Relation

对 normalized exact Signal embedding，在 batch 内以 RGBNT201 **物理 camera
ID** 选择：

\[
p_0(q)=\arg\min_{p:y_p=y_q,c_p\ne c_q}s(z_0^q,z_0^p),\qquad
n_0(q)=\arg\max_{n:y_n\ne y_q}s(z_0^q,z_0^n).
\]

negative 不要求不同 camera；无异 camera positive 的 query 直接排除。所有输出
都在同一 `(q,p0,n0)` 计算：

\[
m_e=s(z_e^q,z_e^{p_0})-s(z_e^q,z_e^{n_0}),
\ e\in\{0,C,T,M,fused\}.
\]

每 batch 至多 B 条 relation，不枚举 B³ triplets，不允许每支自行更换 hard pair。

### SATR Contract

对 receiver `e`，另两支为 `j,k`：

\[
\bar t_e=\min(\operatorname{sg}(m_j),\operatorname{sg}(m_k)),
\]

\[
a_e=[\bar t_e>0]\land
[\bar t_e\ge\max(\operatorname{sg}(m_0),
\operatorname{sg}(m_e))+0.05].
\]

teacher 与 mask 全 detached，live receiver margin 只进入：

\[
L_{SATR}^{e}=
\frac{\sum_q a_e(q)\frac12
\max(0,\bar t_e(q)-m_e(q))^2}
{\max(1,\sum_q a_e(q))}.
\]

空 mask 精确为0；不切换 single peer。三支轮流做 receiver，但方向由每条
relation 的交集决定。

### Fused Safety

\[
a_p=[\operatorname{sg}(m_0)\ge0.30],
\]

\[
L_{protect}=
\frac{\sum_q a_p(q)
\max(0,\operatorname{sg}(m_0(q))-0.02-m_{fused}(q))^2}
{\max(1,\sum_q a_p(q))}.
\]

它只限制 fused 对 strong-Signal relation 的退化；不会保证 fused gain，也不约束
三个 private embeddings 相互相似。

### Frozen Objective and Constants

复用既有 ReID 权重：fused ID/triplet `0.25/1.0`；每 branch 与 residual
ID/triplet `1/12/0.25`；triplet margin `0.3`；label smoothing `0.1`。

\[
L=L_{ReID}+1.0\sum_eL_{SATR}^e+0.25L_{protect}.
\]

完整候选披露：

- `delta_r=0.05`：唯一执行值；
- `gamma_p=0.10`：仅 fold0 sanity，coverage=100%，作为饱和值拒绝；
- `gamma_p=0.30`：唯一 replacement，在三折冻结；
- `epsilon_p=0.02, lambda_r=1.0, lambda_p=0.25`：未执行替代候选；
- 所有决定均 optimizer0/train-only；Q1 endpoint/dev/official 从未参与。

### Fixed-initial Activity Gate

在三个 V12 fold 的相同 seed42、前八个 B64/K8 source-training batches 上，
optimizer0 重放 initial model。每 fold/receiver 的

\[
coverage_e=\frac{\sum_q a_e(q)}{\#\ valid\ cross-camera\ queries}
\]

必须在 `[0.5%,25%]`。已登记初态值：

- fold0 CNN/T/M `1.56/3.13/4.69%`；
- fold1 `3.13/1.56/14.06%`；
- fold2 `4.17/1.39/5.56%`。

M0 必须在 clean code 上复现该区间。训练期 coverage 逐 epoch 记录为诊断，可
随修复下降；它不参与 scientific pass/fail，也不触发 fallback。

### Fair Q1 Endpoint Pair

每个 fold 从同一完整 state SHA 构建：

- `SATR`：完整 objective；
- `matched no-SATR`：相同 objective 但 `lambda_r=lambda_p=0`。

两端点共享同一94 source identities、trainable names、B64/K8、20 epochs、
AdamW LR `3.5e-4`、WD `1e-4`、AMP256、final epoch only。每个端点前重置
seed42并重建 loader；receipt 记录：

1. initial full-state SHA；
2. trainable-name SHA；
3. ordered sample-path + sampler-index SHA；
4. augmentation seed contract；
5. 前八个 post-transform RGB/NIR/TIR tensor SHA。

任一 paired hash 不同则 integrity FAIL，不解释性能。两个端点在同一47 unseen
identities paired evaluation，dev0/official0。

### Inference

relation path 全部删除：

\[
z_{fused}=[z_0,\|z_0\|\operatorname{Normalize}
(\operatorname{Concat}(r_C,r_T,r_M))].
\]

输出 exact baseline、fused、CNN、Transformer、Mamba；无 Router、exchange、
learned alpha 或额外 FLOPs。branch repair 能否变成 fused gain 不作假设，只由
Q1/D1 判定。

### Claim-driven Validation

**Q1 claim：SATR 对 unseen identities 产生可归因的三方互促。**

- 每 fold fused SATR-minus-noSATR mAP `>=0`；
- 聚合 fused gain `>=+1.0 mAP`，21-identity bootstrap lower bound `>0`；
- aggregate CNN/T/M gain 全部 `>0`，每 fold 至少两支 `>0`；
- SATR fused 严格高于 SATR CNN/T/M；
- fixed-initial activity 与 paired-draw integrity PASS。

Q1 任一门失败即封存 V16，不访问 dev。

**D1 claim：training-only collaboration 形成无动态模块的部署增益。**

仅 Q1 全门通过后，从 V8 all-fit Phase-A 启动同一 SATR 配方20 epoch、final-only
dev。fused 必须 `>=65 mAP`，严格高于 exact Signal、V8 Phase-B `58.4050` 和
V16 CNN/T/M；Rank-1 严格高于三分支；official0。

### Failure Boundaries

- initial activity 空/饱和：FAIL，无阈值重选；
- training coverage 下降：只记录，不误判为失败；
- branch 改善未转 fused：Q1/D1 fused gate 否定，不加 Router/alpha；
- 仅一支受益：branch gate 否定“互促”；
- paired draw/hash 不同：integrity FAIL，先修执行，不读取 dev。

### Compute

- M0：15–25分钟；
- Q1 双端点三折：70–110分钟；
- conditional D1：20–30分钟 + dev约1分钟；
- 晋级前约2 GPU-hours，单 RTX3090，seed42。

