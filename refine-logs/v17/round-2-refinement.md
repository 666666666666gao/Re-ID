# V17 Round 2 Refinement

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

- 原瓶颈仍是“关系层互补无法被单样本 Router 或稀疏/hidden exchange 转成
  held-out retrieval gain”。
- 修订只冻结 loss scalar 和 receipt，不改变问题、模型拓扑或 no-reranking 推理。
- 没有采纳任何会引入新 backbone、test graph、fallback 或实验菜单的建议。

## Simplicity Check

- 唯一新模块仍是一个共享 `TriadicCorrection`。
- 唯一主创新仍是 heterogeneous relation-envelope distillation。
- `L_env` 的四级权重被写成一个无歧义标量公式；不增加新 loss。
- teacher-source distribution 只是一条同run receipt，不新增门槛或实验。

## Changes Made

### 1. 冻结唯一的 loss scalar 约定

对每个 output，正负 envelope 各占一半；在全部 output 中，fused 占一半，三个
corrected branches 合计占一半：

```math
E(u)=\frac12E^+(u)+\frac12E^-(u),
```

```math
L_{env}=\frac12E(f)+\frac16\sum_{e\in\{C,T,M\}}E(h_e).
```

因此 `L_env` 内正/负总权各0.5，fused/branches总权各0.5；外部系数固定1.0。

### 2. 增加 teacher-source 描述性回执

每折记录 `t+` 的 unique argmax 和 `t-` 的 unique argmin 分别由 C/T/M 提供的
pair count、fraction与tie fraction。它不新增post-hoc门；三方claim仍由三个
corrected branch aggregate gain均>0和unique AP wins来裁决。若某支teacher占比
很低，结果表述必须降级，不能隐藏。

### 3. 澄清为什么不受 query-level Oracle 上限约束

V8 branch Oracle 对每个query只能选择一个完整专家输出；DTRED envelope允许同一
query的某个positive pair继承CNN，而另一个negative pair继承Transformer或Mamba，
然后由student学习一个统一几何。因此`64.785 mAP`不是数学上限，只是风险提示。
能否转成一致embedding仍由Q1/D1实证决定。

## Revised Proposal

# Dense Triadic Relation Envelope Distillation（DTRED）

## Technical Gap

当前系统已有强 Signal/CLIP anchor 和三种异构 residual experts。证据表明它们在
不同query上互补，但固定融合几乎不增益，Router不能从单样本统计预测图库相关的
最优专家，hidden exchange会破坏检索，而V16稀疏teaching根本没有稳定覆盖三支。
所需的最小机制是：训练时直接从三专家的pairwise geometry构造稠密、单向的关系
包络，把图库相关互补压缩进测试时可独立计算的embedding。

Test-time rank/graph consensus虽直接，却属于reranking并改变协议，明确排除。

## Thesis and Novelty Boundary

**Thesis**：DTRED把正pair的三专家最大相似度作为student下界，把负pair的三专家
最小相似度作为student上界；单向约束允许student超过teacher，并同时训练CNN、
Transformer、Mamba corrected outputs与fused output。

标准hard mining从同一embedding中选困难样本；标准relational KD通常回归一个
teacher距离矩阵。DTRED的机制边界是：

1. 三个异构inductive-bias experts共同定义label-conditioned relation envelope；
2. one-sided constraint只罚未达到包络，不罚超越；
3. relation被压缩到no-reranking single-sample embedding；
4. complete-path identity-OOF gate要求三个receiver共同获益。

max/min运算、Signal prefix和标准ID/triplet均不单独宣称创新。

## Architecture

冻结exact Signal和V8/V12 pretrained-tail experts。三支输出
`r_C,r_T,r_M in R^1536`。唯一新模块：

```math
q_e=P_{in}(LN(r_e))\in R^{256},\qquad
p_e=q_j\odot q_k,
```

```math
\delta_e=P_e(MLP([q_e,p_e,\bar q])),\qquad
h_e=Normalize(r_e+\delta_e).
```

`P_in`和MLP共享，三个output projections保留receiver身份。随后：

```math
c=Normalize([h_C,h_T,h_M]),
```

```math
z_f=Concat(z_{Signal},||z_{Signal}||c),
```

```math
z_e=Concat(z_{Signal},||z_{Signal}||h_e).
```

输出`baseline_only/fused/cnn/transformer/mamba`。query/gallery独立forward，标准
L2 distance，`RERANKING=false`，无sample alpha和runtime fallback。

## Source-only Relation Envelope

对Q1 fold `k`，teacher和gradient只接触该fold的94个source-training identities。
47个held-out identity labels只在final epoch完成后用于一次train-only资格评价。

对source batch内合法pair：

```math
s^e_{ij}=\langle\hat r^e_i,\hat r^e_j\rangle,
```

```math
t^+_{ij}=\max_e s^e_{ij},\quad y_i=y_j,c_i\ne c_j,
```

```math
t^-_{ij}=\min_e s^e_{ij},\quad y_i\ne y_j.
```

teacher全部detach。定义：

```math
E^+(u)=Mean_{P^+}[ReLU(t^+_{ij}-s^u_{ij})^2],
```

```math
E^-(u)=Mean_{P^-}[ReLU(s^u_{ij}-t^-_{ij})^2],
```

```math
E(u)=\frac12E^+(u)+\frac12E^-(u),
```

```math
L_{env}=\frac12E(f)+\frac16(E(h_C)+E(h_T)+E(h_M)).
```

B64/K8 formal batch必须同时存在合法positive/negative pair；否则M0直接失败，不
添加fallback。正负集合分开求mean，避免negative数量主导。student若已优于某一
包络，hinge为0。

## Signal Protection and Total Objective

只对fused复用已formal执行的V16保护常数：

```math
L_{safe}=ReLU(m_S-0.02-m_f),\quad m_S\ge0.30.
```

总目标冻结为：

```math
L=L^f_{id+tri}+\frac13\sum_eL^{h_e}_{id+tri}
+1.0L_{env}+0.25L_{safe}.
```

ID label smoothing0.1；triplet对L2-normalized embedding；Signal、三expert和
teacher均冻结。无branch safety subsystem。

## Paired Q1 Contract

DTRED与weight0端点的initial state、trainable names、seed、sampler indices、前8
个增强后RGB/NIR/TIR tensor SHA、optimizer、batch和step完全一致；唯一差别是
`L_env`外部系数1或0。weight0是Q1因果资格对照，不是成功后的论文消融。

每折同run receipt记录：

- source-training teacher `t+` unique argmax与`t-` unique argmin的C/T/M count、
  fraction、tie fraction；
- fused/C/T/M的`E+`、`E-`、总violation与satisfaction；
- DTRED相对weight0的正负violation差；
- corrected branch pairwise cosine、mAP和unique AP wins；
- source/heldout identity lists、所有核心文件/checkpoint SHA、dev0/official0。

teacher source分布是透明描述，不触发新参数调整；scientific gate不因观察到的
占比而修改。

## Minimal Execution Plan

### M0 engineering gate

1. TDD覆盖one-sided方向、正负独立mean、固定scalar、detach和output prefix；
2. real B64/K8 8 steps，所有trainable tensor有有限非零gradient，0 overflow；
3. fixed-batch100-step floor-aware overfit；
4. 三fold各首8个hash-bound batch同时有合法正/负pair，fused/C/T/M envelope路径
   均贡献gradient；
5. exact Signal/frozen expert SHA不变，dev0/official0。

### Q1 identity-OOF gate

三fold各20 epoch，seed42，final-only paired endpoints。全部满足才授权D1：

- 每fold fused mAP gain>=0；
- 571-query weighted fused gain>=+1.0 mAP；
- identity-cluster bootstrap10k 95% LB>0；
- CNN/Transformer/Mamba aggregate gain均>0；
- DTRED相对weight0的fused positive和negative violation均下降；
- paired/integrity/frozen/dev0/official0全部通过。

unique AP wins和teacher source分布解释triadic claim，但不形成可调门。

### D1 no-reranking main dev

仅Q1全门通过后：141 fit identities、seed42、20 epoch、final-only、一次30-dev。
fused必须`>=65 mAP`并严格超过exact Signal、V8 Phase-B及三个corrected branches。
失败即封存，不做width/loss/LR/epoch/checkpoint扫描。

## Failure and Claim Boundaries

- Q1 envelope violation不改善：机制未发生；即使偶然mAP正也不归因DTRED。
- 任一branch aggregate gain<=0：不支持三方互促，不进入D1。
- D1<65或不严格胜comparator：无部署/SOTA claim，不加runtime fallback。
- Q1 metrics只称train-only mechanism qualification，不与30-dev 65门横比。
- D1成功前不做论文消融、多种子或official；成功后再补最小必要性表与公平SOTA表。

## Compute

- M0约5--8分钟；Q1约35--50分钟；条件D1约25--35分钟。
- D1前约1 GPU-hour，单3090可承载，无额外数据或标注。
