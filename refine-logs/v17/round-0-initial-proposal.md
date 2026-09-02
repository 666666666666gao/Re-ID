# 研究方案：V17 Dense Triadic Relation Envelope Distillation（DTRED）

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

## Technical Gap

现有证据把失败点定位得很窄：不是 GPU、未训练完或三个架构没有容量，而是
“如何选择/交换”的监督接口错误。V8 residual-only Oracle 表明专家在不同查询上
各有优势；然而最优专家依赖查询相对图库的检索关系，单张图的质量统计不足以
预测它。V15/V16 进一步表明，把互补性压成 hidden delta 或依赖极少数 hard mask
也不稳定。

因此最小缺失机制不是更深的第四个融合网络，而是一个**训练期的稠密关系教师**：
对每个同身份/异身份 pair，直接从 CNN、Transformer、Mamba 三种检索关系中取
能够保留身份的包络，再把该关系蒸馏进每个样本独立可计算的 corrected embedding。
这样训练时可以使用邻域/关系信息，测试时仍只有普通单样本 embedding 和欧氏距离。

### 两条候选路线

- **Route A：test-time cross-expert neighborhood consensus**。直接融合三个专家
  的 query-gallery 排名，最贴近 Oracle，工程上也最简单；但它改变了冻结的
  `RERANKING=false` 协议，不能与当前 65 mAP 门和无重排序 SOTA 公平比较，故不选。
- **Route B：train-time relation-envelope distillation**。训练时使用真实 fit identity
  关系构造三专家包络教师，将其蒸馏为单样本 corrected embedding；推理仍是一次
  forward + L2 distance。它直接针对证据中的“关系可见、单样本 Router 不可学”
  瓶颈，且不改变评估协议，故选择 Route B。

## Method Thesis

- **One-sentence thesis**：用 label-conditioned 的 CNN/Transformer/Mamba 关系
  包络作为稠密教师，训练一个共享低秩 triadic correction，使三个专家在保留
  私有表示的同时吸收其他两支的互补检索关系，并将其压成 no-reranking 的固定
  embedding。
- **Smallest adequate intervention**：冻结 exact Signal 和 V8/V12 专家，只新增
  一个共享低秩 triadic correction；删除 Router、HFER、CRDE、SATR hard mask、
  learned sample alpha、memory bank 和测试时图库图。
- **Foundation-model-era leverage**：CLIP/Signal 提供强且冻结的语义锚；蒸馏不是
  从额外模型复制 logits，而是把三种异构 inductive bias 的 pairwise relation
  envelope 压回一个样本级表示。这比增加新 foundation backbone 更贴合瓶颈。

## Contribution Focus

- **Dominant contribution**：Dense Triadic Relation Envelope Distillation（DTRED）——
  将异构专家的 label-conditioned 最优 pair relation 变成稠密教师，并同时监督
  三个 corrected expert 与 fused retrieval geometry。
- **Supporting contribution**：Signal-anchored one-sided protection，只在 exact
  Signal 对当前 hard pair 已可靠时阻止 fused relation 变差；它是安全约束，不单列
  为论文主贡献。
- **Explicit non-contributions**：不把 V8 预训练三专家、标准 ID/triplet、共享
  geometric augmentation 或 exact prefix 单独宣称为新贡献；不把 no-reranking
  当创新。

## Proposed Method

### Complexity Budget

- **Frozen/reused**：exact Signal 3072D embedding；V8 pretrained-tail CNN、
  Transformer、Mamba residual experts；现有同步三模态数据管线。
- **New trainable component**：一个共享 `TriadicCorrection`，由共享 1536->256
  投影、peer-intersection MLP 和三个轻量 output projections 组成；训练期 classifiers
  不计入推理组件。
- **Intentionally excluded**：新 Router、额外 attention stage、第二个 fusion head、
  EMA teacher、DINO/SAM、test-time graph、可学习样本级 alpha。

### System Overview

```text
RGB / NIR / TIR（共享几何增强）
        ↓
frozen exact Signal + frozen pretrained-tail experts
        ├─ r_C：CNN local/high-frequency residual, 1536D
        ├─ r_T：Transformer global-relation residual, 1536D
        └─ r_M：Mamba long-range residual, 1536D
                    ↓
     shared low-rank projection q_C, q_T, q_M
                    ↓
peer intersection p_e = q_j ⊙ q_k
                    ↓
δ_e = P_e([q_e, p_e, mean(q_C,q_T,q_M)])
                    ↓
h_e = Normalize(r_e + δ_e)
        ├─ corrected CNN branch
        ├─ corrected Transformer branch
        └─ corrected Mamba branch
                    ↓
c = Normalize(Concat(h_C,h_T,h_M))
                    ↓
z_fused = Concat(z_Signal, ||z_Signal|| · c)
```

推理输出仍是固定 embedding；query 与 gallery 均独立 forward，评价继续使用标准
L2-normalization、欧氏距离、mAP/CMC，`RERANKING=false`。

### Core Mechanism：Relation Envelope Teacher

对 batch 内合法 pair `(i,j)`，每个冻结专家给出余弦相似度：

```math
s^e_{ij}=\langle \hat r^e_i,\hat r^e_j\rangle,
\quad e\in\{C,T,M\}.
```

使用真实 fit identity 只在训练期构造包络教师：

```math
t_{ij}=\begin{cases}
\max_e s^e_{ij}, & y_i=y_j,\; camera_i\ne camera_j,\\
\min_e s^e_{ij}, & y_i\ne y_j.
\end{cases}
```

正 pair 继承三专家中最能拉近同身份者，负 pair 继承最能分离不同身份者。它不是
模型输出伪装成 GT：identity/camera 来自 RGBNT201，expert relation 只构成明确标注
的训练教师。teacher 全部 `detach`。

corrected expert 和 fused residual 的相似度共同拟合教师：

```math
L_{env}=\frac{1}{|P|}\sum_{(i,j)\in P}
\left[\ell(s^f_{ij},t_{ij})+
\frac{1}{3}\sum_e\ell(s^{h_e}_{ij},t_{ij})\right],
```

其中 `ell` 为 Smooth-L1。每个合法 pair 都贡献梯度，不再依赖 V16 的稀疏 mask。
三个 corrected branches 仍以各自 `r_e` 为 residual anchor，因此共享关系不会直接
覆盖其私有架构特征。

### Signal-Anchored Protection

在 exact Signal 已提供可靠 batch-hard relation 时，只施加 one-sided fused margin
保护：

```math
L_{safe}=\max(0,m_S-0.02-m_f),\quad m_S\ge0.30.
```

`0.30/0.02` 复用已登记且 formal M0 实际执行过的 V16 protection 常数，不重新
扫描。它只约束 fused，不参与包络教师选择，也不作为 runtime fallback。

### Total Objective

```math
L=L_{id+tri}^{fused}
+\frac{1}{3}\sum_e L_{id+tri}^{h_e}
+1.0L_{env}+0.25L_{safe}.
```

ID 使用 label smoothing 0.1；triplet 在 L2-normalized embedding 上计算。所有
冻结 expert 和 Signal 参数不更新。matched weight0 endpoint 只把 `L_env` 权重置0，
其他初态、数据、优化器和训练步完全一致，用于 train-only 因果资格门，不视为
成功后的论文消融。

### Integration and Inference

- Q1：每个 V12 complete-path fold 用94个 fit identities训练 correction，只在47个
  heldout identities中的 cross-camera eligible records 评价；三折都不访问30-dev。
- D1：仅在 Q1 通过后，复用 V8 all-fit expert checkpoint，在全部141 fit identities
  训练同一 correction，固定最终 epoch，唯一一次评价30-dev。
- 输出必须同时包含 `baseline_only`、`fused`、`cnn`、`transformer`、`mamba`；
  exact Signal prefix 必须 bit-exact。没有 fused 低于 baseline 时动态切回 baseline
  的 runtime 分支。

## Training Plan

1. **M0 engineering gate**：TDD、exact prefix/frozen SHA、B64/K8 8-step capacity、
   100-step fixed-batch overfit；验证 envelope pair coverage 在每个 fold 对正负 pair
   均非零且三个 output projections 均有梯度。coverage 使用完整 sampler indices 与
   增强后 RGB/NIR/TIR tensor SHA，不设可调 activity 阈值。
2. **Q1 identity-OOF qualification**：三折各20 epoch，matched DTRED/weight0；终点
   固定，不做 checkpoint selection。要求每折 fused gain>=0，query-weighted aggregate
   >=+1.0 mAP，10k identity-cluster bootstrap 95% LB>0，三个 corrected expert
   aggregate gain均>0。
3. **D1 main dev**：Q1全门通过才运行；all-fit seed42 20 epoch、final-only、一次
   no-reranking dev。要求 fused>=65 mAP，严格超过 Signal、V8 Phase-B和三个分支。
4. **之后**：只有 D1 主指标通过，才计划消融、官方test和同协议SOTA表。

## Failure Modes and Diagnostics

- **Envelope不可学习**：Q1 aggregate或bootstrap gate失败；封存V17，不调温度、
  hidden width、loss权重或epoch。
- **三支被拉成同质表示**：记录 corrected branch pairwise cosine和独有AP wins；若
  三个 branch aggregate gain不能全正，三方互促claim失败并停止D1。
- **Fused伤害Signal**：Q1 per-fold non-inferiority或D1 strict comparator gate失败；
  不增加runtime fallback。
- **协议漂移**：receipt强制 `reranking=false`、dev/official access计数、fold identity
  disjoint、source/config/runner/checkpoint SHA。

## Novelty and Elegance Argument

UniCat 暴露 modality laziness；DeMo/UGG-ReID 用 routing 或 uncertainty；MambaPro
用 CLIP+Mamba 聚合；Signal/STMI 使用 token/graph relation。DTRED 的区别不在再加
一层 fusion，而在**以三种异构架构对同一 pair 的互补关系构造 label-conditioned
envelope，并把图库相关的关系优势蒸馏为无需图库交互的单样本 embedding**。它将
V8 已观测到但无法被 sample-local Router 捕获的 Oracle complementarity 转换成
稠密训练信号，方法接口窄、无额外 backbone、无测试时图和无参数菜单。

## Claim-Driven Validation Sketch

### Claim 1：DTRED 能把 identity-OOF 三专家关系互补转成稳定 fused gain

- **Minimal experiment**：三折 complete-path Q1 matched DTRED vs weight0。
- **Metrics**：每折/聚合 mAP、identity-bootstrap LB、三个 corrected branch gain。
- **Decisive evidence**：每折 fused非负、aggregate>=+1.0、LB>0、三branch均正。

### Claim 2：稠密关系蒸馏能生成 no-reranking 的部署增益

- **Minimal experiment**：唯一 seed42 D1 final-only dev。
- **Comparators**：exact Signal、V8 Phase-B、corrected CNN/T/M。
- **Decisive evidence**：fused>=65 mAP且严格胜全部comparators；否则不支持。

## Experiment Handoff Inputs

- **Must-prove claims**：identity-OOF稳定互促；no-reranking deployable gain。
- **Must-run ablations**：成功前无论文消融；Q1 weight0只作资格门。
- **Critical dataset/metrics**：RGBNT201 141-fit/30-dev；mAP、Rank-1、identity-cluster
  bootstrap；official test仅在D1通过后。
- **Highest-risk assumptions**：pairwise envelope是否能由单样本 correction泛化；
  同一 envelope监督是否会抹平专家私有结构。

## Compute & Timeline Estimate

- M0：约5--8分钟 RTX3090。
- Q1：约35--50分钟 RTX3090（三折matched endpoint）。
- D1：Q1通过后约20--30分钟训练加3--5分钟dev。
- 总授权上限：D1前约1 GPU-hour；无额外数据或人工标注。
