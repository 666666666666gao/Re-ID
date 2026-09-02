# V17 Round 1 Refinement

## Problem Anchor

- **Bottom-line problem**：在固定 RGBNT201 `141-fit / 30-dev`、seed42、单张
  RTX3090、`RERANKING=false` 协议下，让共享 Signal/CLIP 语义主干上的 CNN、
  Transformer、Mamba 三个异构专家形成可部署的深度协同，并把当前
  V8 Phase-B 的 `58.4050 mAP / 59.3939 Rank-1` 提升到至少 `65 mAP`。
- **Must-solve bottleneck**：V8 已证明三个专家存在显著 query-level Oracle
  互补，但 V12--V14 的 sample-local Router 无法跨身份预测最优专家，V15 的
  hidden exchange 伤害检索，V16 的 sparse two-peer teaching 在 Transformer
  三折均为零活动。必须把“专家互补”转成稠密、身份隔离、能泛化到单样本
  embedding 的监督，而不是继续调 Router 或稀疏阈值。
- **Non-goals**：不做 test-time graph/rank fusion 或 k-reciprocal reranking；不
  新增 DINO/SAM/VLM/完整 backbone；不重跑 baseline；不进行多种子、参数扫描、
  消融或 official test，除非冻结的主实验门先通过。
- **Constraints**：所有训练与评估只在远端 RTX3090；物理 batch B64/K8；只用
  seed42；保留 exact 3072D Signal output；V12 complete-path identity-OOF 仅用于
  train-only 资格门；30-dev 最多一次冻结终点评估；官方 test 在 dev 晋级前为0。
- **Success condition**：train-only 三折资格门证明 DTRED 相对 matched weight0
  在每折不劣、聚合 fused 至少 `+1.0 mAP`、identity-bootstrap 95% lower bound
  大于0且三个 corrected expert 聚合均为正；随后唯一 D1 no-reranking dev 达到
  `>=65 mAP`，并严格超过 exact Signal、V8 Phase-B 和三个 corrected branches。

## Anchor Check

- **Original bottleneck**：异构专家已经有 query-level 互补，但样本局部 Router、
  hidden exchange 和稀疏 hard teaching 都不能把它稳定转成 held-out identity 检索
  增益。
- **Why this still addresses it**：修订后仍只改变“互补关系如何成为训练监督”，
  不增加新 backbone，不让测试样本读取图库，也不改评估协议。
- **Reviewer suggestions rejected as drift**：无。审阅者没有要求 DINO、SAM、
  graph inference 或 reranking。

## Simplicity Check

- **Dominant contribution**：异构三专家关系包络的单向约束蒸馏。
- **Components removed/merged**：删除精确 Smooth-L1 target regression；正负关系
  改成两个单向 hinge，并在内部各自归一后相加。仍只有一个
  `TriadicCorrection`。
- **Complexity rejected**：不增加 EMA teacher、memory bank、第二 fusion head、
  branch safety module 或 runtime fallback。
- **Why smallest adequate**：只修复会“惩罚优于教师”和“负 pair 淹没正 pair”的
  两个直接问题，不改变系统图。

## Changes Made

### 1. 双向回归改为单向 relation envelope

- **Reviewer said**：max/min target 未必能由一个 cosine embedding 精确实现，
  Smooth-L1 还会把更好的 student 拉回。
- **Action**：positive 仅在 corrected similarity 低于三专家最大值时罚；negative
  仅在 corrected similarity 高于三专家最小值时罚。
- **Impact**：teacher 从精确数值目标变成“至少达到当前三专家最好关系”的下界/
  上界，不再抑制超越 teacher 的 student。

### 2. 正负 pair 分开归一

- **Reviewer said**：B64/K8 中 negative 数量远多于 cross-camera positive。
- **Action**：`mean(L_pos)+mean(L_neg)`，两个集合独立求均值；没有合法 pair 时
  M0 直接失败，不添加 loss fallback。
- **Impact**：dense teacher 不再等价于只优化负 pair。

### 3. 封死 teacher leakage 与机制回执

- **Action**：每个 Q1 fold 的 envelope 只从该 fold source-training identities 的
  frozen expert relations构造；held-out identity labels只用于终点评价，不进入梯度、
  teacher或模型选择。Q1同时记录正/负 envelope violation、satisfaction、三branch
  cosine、unique AP wins和weight0差值。
- **Impact**：能区分“loss被调用”与“关系包络真的被学习”，同时保持Q1是一个
  experiment block。

## Revised Proposal

# 研究方案：Dense Triadic Relation Envelope Distillation（DTRED）

## Technical Gap

V8 的 branch Oracle `64.785 mAP` 和 residual Oracle `63.481 mAP` 只证明不同
查询由不同专家处理时存在上限，不保证一个固定融合自然能达到该上限；事实上 V8
固定融合只有 `58.097 mAP`，V12--V14 Router、V15 exchange、V16 sparse repair均
未通过身份隔离资格门。关键缺失机制是：用稠密 pair relation 教会一个单样本
embedding 吸收三种专家的互补几何，而不是继续预测离散 expert id。

### Route decision

- Test-time rank/graph consensus最接近Oracle，但属于reranking，违反固定协议。
- Train-time heterogeneous relation-envelope distillation把同样的关系优势压回普通
  embedding，推理仍是独立forward与L2 distance。选择后者。

## Method Thesis and Contribution

**Thesis**：对每个 source-training pair，CNN/Transformer/Mamba 各给出一种冻结
关系；DTRED把正pair中最强的同身份相似度作为下界、负pair中最强的异身份分离
作为上界，以单向约束同时训练三个 corrected experts 和 fused embedding。

**Dominant contribution**不是max/min hard mining本身，而是：

> 在 complete-path identity-OOF 条件下，将 CNN/Transformer/Mamba 异构专家的
> pairwise relation envelope 蒸馏为 no-reranking、single-sample embedding，并用
> corrected branch gain验证三方关系传递。

Signal protection只是fused安全约束。V8专家、ID/triplet、exact prefix和共享几何
增强均不单列为新贡献。

## Architecture and Complexity

冻结 exact Signal 与 V8/V12 pretrained-tail experts。三支各输出
`r_C,r_T,r_M in R^1536`。唯一新模块 `TriadicCorrection` 包含：

```math
q_e=P_{in}(LN(r_e))\in R^{256},
```

```math
p_e=q_j\odot q_k,
```

```math
\delta_e=P_e(MLP([q_e,p_e,\bar q])),\qquad
h_e=Normalize(r_e+\delta_e).
```

`P_in`与MLP共享，只有三个output projection区分receiver。没有attention stage、
Router或第二fusion head。最终：

```math
c=Normalize([h_C,h_T,h_M]),
```

```math
z_f=Concat(z_{Signal},||z_{Signal}||c).
```

分支输出为`Concat(z_Signal,||z_Signal||h_e)`。`baseline_only`仍是bit-exact
3072D Signal。query/gallery独立forward，`RERANKING=false`。

## One-sided Relation Envelope

在每个Q1 fold中，teacher只读取该fold source-training identities。对合法pair：

```math
s^e_{ij}=\langle \hat r^e_i,\hat r^e_j\rangle,
```

```math
t^+_{ij}=\max_e s^e_{ij}\quad(y_i=y_j,c_i\ne c_j),
```

```math
t^-_{ij}=\min_e s^e_{ij}\quad(y_i\ne y_j).
```

所有teacher tensor均detach。对任一corrected output `u`：

```math
L^+_{env}(u)=Mean_{P^+}[ReLU(t^+_{ij}-s^u_{ij})^2],
```

```math
L^-_{env}(u)=Mean_{P^-}[ReLU(s^u_{ij}-t^-_{ij})^2].
```

正负集合独立归一；B64/K8每个formal batch必须同时含合法正/负pair，否则M0失败。
总包络损失：

```math
L_{env}=L^+_{env}(f)+L^-_{env}(f)
+\frac13\sum_e[L^+_{env}(h_e)+L^-_{env}(h_e)].
```

student若比teacher更好，hinge为0，不被拉回。包络不需要由一个embedding精确
回归，只要求减少可实现的violation。

## Signal Protection and Total Loss

复用已formal执行的V16常数，只在Signal hard relation可靠时保护fused：

```math
L_{safe}=ReLU(m_S-0.02-m_f),\quad m_S\ge0.30.
```

无branch protection、无runtime fallback。总损失：

```math
L=L^{f}_{id+tri}+\frac13\sum_e L^{h_e}_{id+tri}
+1.0L_{env}+0.25L_{safe}.
```

ID label smoothing0.1；triplet使用L2-normalized embedding。teacher、Signal、三
expert均冻结。

## Q1 Teacher Boundary

对fold `k`：

1. 仅在94个source-training identities上训练`TriadicCorrection`并构造envelope；
2. 不将47个held-out identity的label、feature relation或metric用于teacher、梯度、
   threshold、epoch或checkpoint选择；
3. final epoch固定后，才在held-out cross-camera records计算mAP/Rank-1；
4. DTRED与weight0从同一initial state、相同sampler indices、增强后tensor SHA、
   optimizer、step和seed开始；weight0只去掉`L_env`，其余不变。

这与V12完整路径隔离一致，但Q1 held-out metric只作为train-only mechanism
qualification，绝不与30-dev的65门横向比较。

## Mechanism Receipt

Q1在同一终点额外记录，不增加实验配置：

- positive/negative `L_env` 与 violation magnitude；
- positive satisfaction `s_corr>=t+`、negative satisfaction `s_corr<=t-`；
- fused及CNN/T/M分别相对weight0的变化；
- corrected branch pairwise cosine；
- held-out fixed branch mAP与unique AP wins。

DTRED必须相对weight0降低正负violation；否则即使mAP偶然提高，也不能归因为
relation-envelope机制。

## Training and Gates

### M0

- 公共接缝TDD；exact prefix/frozen SHA；
- real B64/K8 8-step capacity、0 overflow、所有trainable tensor有gradient；
- 100-step fixed-batch floor-aware overfit；
- 每fold首8个hash-bound batch均有正/负合法pair，且fused/C/T/M的正负envelope
  路径均实际贡献非零gradient。

M0没有可调coverage阈值，不使用dev/official。

### Q1

三fold x20 epoch，seed42，final-only matched endpoint。全部满足才授权D1：

- 每fold fused mAP gain >=0；
- 571-query weighted aggregate fused gain >=+1.0 mAP；
- identity-cluster bootstrap 10k 95% lower bound >0；
- CNN/Transformer/Mamba aggregate gains均>0；
- DTRED相对weight0的positive与negative violation均下降；
- frozen SHA、paired hash、dev0、official0全部通过。

### D1

Q1全门通过后，在141 fit identities训练all-fit correction 20 epoch，final-only唯一
一次30-dev，no-reranking。fused必须`>=65 mAP`并严格超过exact Signal、V8
Phase-B和三个corrected branches。失败即封存；不做checkpoint/LR/width/loss扫描。

## Failure Modes

- **非度量envelope不可满足**：机制receipt不改善或Q1失败，直接封存；不改成双向
  回归、不加temperature。
- **branch同质化**：三branch cosine上升且unique wins消失或任一aggregate gain<=0，
  三方互促claim失败。
- **只提高fit不提高heldout**：Q1 per-fold/bootstrap gate阻止D1。
- **只靠Signal保底**：fused必须严格胜V8/branches，且没有runtime fallback。

## Novelty Boundary

标准hard mining从一个embedding里挑困难样本；标准relational KD通常让student
拟合一个teacher的距离矩阵。DTRED使用三个具有不同inductive bias的冻结专家，按
真实pair标签形成方向相反的relation envelope，并以one-sided constraint允许student
超越每个teacher；同时要求三个corrected receivers在complete-path identity-OOF上
共同获益。创新边界是**heterogeneous relation transfer + deployable compression +
identity-isolated qualification**的整体，不把max/min运算单独包装成新算法。

## Claim-Driven Validation

### Claim 1：身份隔离的三方关系传递

单个Q1 matched block；决定性证据是每折fused非负、aggregate>=+1、LB>0、三支
aggregate均正且正负envelope violation同时优于weight0。

### Claim 2：no-reranking部署提升

唯一D1 final-only dev；fused>=65并严格胜Signal、V8和三branch。未达到则不支持。

成功前不做论文消融；成功后才规划最小必要性表和official test。

## Compute and Timeline

- M0：5--8分钟；
- Q1：35--50分钟；
- D1：条件通过后20--30分钟训练+3--5分钟dev；
- D1前总预算约1 GPU-hour，无额外数据/标注。
