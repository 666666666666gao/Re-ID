# Research Proposal: TriFusion V15 — Counterfactual Role-Delta Exchange

## Problem Anchor

- **Bottom-line problem**：在固定 RGBNT201 `141-fit / 30-dev` 协议上，让共享
  CLIP 语义主干上的 CNN、Transformer、Mamba 三个架构专家产生真正的深层
  相互促进，使部署 fused 严格高于 exact Signal 和三个分支，并达到至少
  `65 mAP`；只有同协议结果确实达到公开最佳时才宣称 SOTA。
- **Must-solve bottleneck**：V8 已证明三个专家有 query-level 互补，但选择
  Oracle 也只有 `64.7850 mAP`；V13/V14 证明 sample-local Router 不能稳定预测
  held-out relational utility；V9 又证明在池化残差之后追加 relay 会破坏
  identity-disjoint 泛化。必须生成当前固定分支之外的新身份表示，而不是再次
  调 Router 或在末端堆一个 embedding mixer。
- **Non-goals**：不复现 Signal baseline；不扫描 Router、alpha、LR、epoch、
  temperature 或 exchange 强度；主结果通过前不做消融、多 seed、official test
  或 reranking；不增加 DINO/文本/额外 backbone；不把 diagnostic Oracle 当部署
  结果。
- **Constraints**：所有训练和评估仅在远端单张 RTX 3090 24GB；seed42；共享
  三模态几何增强；真实 B64/K8；exact Signal 3072D 输出逐元素保留；复用 V12
  三个 complete-path identity-OOF Signal/expert checkpoint 和 V8 all-fit Phase-A
  checkpoint；主实验前先过 train-only identity-disjoint 门。
- **Success condition**：新的 stagewise collaboration 在每个完整路径 OOF fold
  上不劣于同 checkpoint 的 no-exchange bank，query-weighted mAP 有预注册正
  增益且 identity-bootstrap 下界为正；随后唯一 all-fit seed42 frozen-dev 结果
  fused `>=65 mAP`，并严格高于 exact Signal、no-exchange bank、CNN、
  Transformer、Mamba。

## Technical Gap

V8 在 CLIP block8 后分叉，三个专家分别通过冻结的 pretrained tail 9/10/11，
并在每层后加入 CNN local、Transformer global、Mamba spatial/cross-modal
adapter。这已经让三个分支形成真实互补，但它们直到最终 residual bank 才
相遇。V8 的 hard Oracle 低于65说明“选哪个专家”本身上限不足；V13/V14 的
失败说明 deployment-local quality feature 不能可靠预测 query-gallery 关系；
V9 则在已经池化、归一化的 residual 上构造正交 relay，并将新向量附加到
Phase-B 后面，结果 beta 接近上限且 dev mAP 降到56.5339。

缺失的机制不是更复杂的 Router，而是：**让一个专家刚产生的结构化增量，在
仍有后续 pretrained CLIP tail 可以重解释它时，成为另外两个专家的受控输入；
同时用 matched no-exchange counterfactual 明确要求通信降低检索风险。**

## Method Thesis

- **One-sentence thesis**：只交换每个专家相对同层 frozen-tail anchor 新产生的
  role delta，并在下一层 pretrained tail 前以有界 residual 注入 peers，再以
  matched no-exchange retrieval regret 训练，可在不改 Signal 和不依赖样本路由
  的情况下，把局部、全局和长程归纳偏置转化为可泛化的联合身份表示。
- **Why smallest adequate**：只增加一个跨三层复用的 Role-Delta Exchange 模块
  和一个训练期 counterfactual loss；复用 V8 三专家、role heads、fusion、数据
  与 checkpoint，不增加 backbone、Router、teacher network 或末端 synthesis。
- **Foundation-model leverage**：CLIP tail 不只是 frozen feature extractor，而是
  作为每轮异构消息之后的 pretrained semantic interpreter；这比在冻结特征后
  再训练 MLP/relay 更符合强 foundation representation 的使用方式。

## Contribution Focus

- **Dominant contribution**：Counterfactual Role-Delta Exchange（CRDE）——在
  三个 pretrained-tail experts 之间进行同步、无 self-edge、source-role-preserving
  的 stagewise delta exchange，并保持 receiver private state 为恒等主路径。
- **Supporting contribution**：Matched Retrieval-Regret Cooperation——每个 batch
  同时产生 exchange-on 与严格 no-exchange 输出，直接惩罚 fused 和三个 receiver
  的 cross-camera batch-hard risk 相对冻结 counterfactual 的退化。
- **Existing foundation contribution retained**：V8 pretrained semantic
  rebranching 让 CNN/Transformer/Mamba 全部继承同一 CLIP tail 语义，同时保留
  exact Signal prefix；V15 不把它冒充新实验结果，但它与 CRDE、regret objective
  共同构成论文的三个方法点。
- **Explicit non-contributions**：不声称首次组合 CNN/Transformer/Mamba；不声称
  HFER、CNN-Transformer exchange 或 Mamba RGBNT 本身首次；不把更多参数、硬
  Oracle 或 synthetic quality response 写成主创新。

## Proposed Method

### Complexity Budget

- **Frozen/reused**：V12 fold-specific 或 V8 all-fit Signal+V8 expert checkpoint、
  frozen CLIP tail blocks、V8 role adapters/heads、shared geometry、ordinary ReID
  evaluator。
- **New trainable components**：一个跨 stage 复用结构的 CRDE ModuleList（每层
  独立参数）和现有输出对应的 BN/classifier heads。
- **Intentionally excluded**：Router、learned alpha/beta、orthogonal projection、
  final synergy MLP、DINO、text、peer distillation、extra backbone。

### System Overview

```text
RGB / NIR / TIR --shared geometry--> frozen Signal
        |                                  |
        |                            exact z0 (3072D)
        v
CLIP block8 anchor A0
        |
        +--> frozen tail9 -> CNN/T/M role adapter -> role deltas D1
        |                                      |
        |                             CRDE exchange E1
        |                                      v
        +--> frozen tail10 <- private state + bounded peer delta
        |            -> CNN/T/M adapter -> D2 -> CRDE E2
        +--> frozen tail11
                     -> CNN/T/M adapter -> D3 -> CRDE E3
                                             |
                 matched final delta to frozen-tail reference
                                             |
                    CNN / Transformer / Mamba residual bank
                                             |
                         [exact z0, equal-energy bank]
```

### Core Mechanism: CRDE

对 stage `l`、source expert `s`，保存 role adapter 前后状态
`X_s^l` 和 `Y_s^l`，只把

\[
D_s^l = Y_s^l-X_s^l
\]

作为通信源。这样共享 CLIP 语义和 receiver 私有路径不会被当作消息重复注入。
每个 source 的 low-rank role mixer 保持其归纳偏置：

- CNN：对 patch delta 做 rank-64 depthwise 2D local mixing；
- Transformer：对 CLS+patch delta 做 rank-64 self-attention，并保留 CLS global
  context；
- Mamba：对 patch delta 做 rank-64 four-direction spatial scan 和已存在的
  RGB/NI/TIR aligned modal scan。

六条 directed peer edges 使用独立 `from_shared[s→t]`。更新同步进行，禁止
self-edge：

\[
\widetilde Y_t^l = Y_t^l +
\sum_{s\ne t}0.25\tanh(\theta_{s\to t}^l)P_{s\to t}M_s(D_s^l).
\]

`theta` 初始化为0，所以新模型 step0 的 exchange-on 与 no-exchange 完全一致；
系数绝对值被固定上界0.25约束。更新后不对完整 receiver state 做 LayerNorm、
不正交化 message，也不重标定到等能量；下一 frozen CLIP tail 自行将小残差重新
映射到 pretrained semantic space。第三层 exchange 直接进入 matched final
delta/role head。

### Supporting Training Signal: Matched Retrieval-Regret Cooperation

每个训练 batch 运行两条共享参数路径：

- `exchange_on`：正常 CRDE；
- `exchange_off`：相同 frozen Signal/expert 输入，六条 edge 系数置零并 detach，
  形成严格 matched counterfactual。

对 fused、CNN、Transformer、Mamba 四个输出使用 L2-normalized embedding，
计算 cross-camera hardest-positive / nearest-negative softplus risk：

\[
R(z)=\frac1{|Q|}\sum_q
\operatorname{softplus}(d^+_{hard}(q)-d^-_{near}(q)).
\]

通信损失为

\[
L_{coop}=\sum_{o\in\{fused,C,T,M\}}
\operatorname{softplus}
\left(\frac{R(z_o^{on})-R(z_o^{off})}{0.05}\right).
\]

总损失只包含现有 V8 ID/triplet 和一个 `L_coop`。`off` 风险不回传；没有
per-query action label、Router target、teacher temperature 或 heldout fold
selection。分类头只用于训练，检索仍使用 pre-BN embedding。

### Inference Path

推理只执行一次 exchange-on：Signal 产生 exact baseline 和 block8 anchor，三个
专家在 tail9/10/11 后交换 role delta，最终输出 baseline-only、fused、CNN、
Transformer、Mamba。正式 fused 沿用 V8 的确定性表示：

\[
z_{fused}=[z_0,\|z_0\|\operatorname{Normalize}
(\operatorname{Concat}(r_C,r_T,r_M))].
\]

它没有 learned Router 或 sample-level scalar，避免 V13/V14 的不可学策略瓶颈；
`baseline_only` 仍逐元素等于 Signal。

### Training Plan

1. **M0 engineering**：synthetic public-seam tests；no-exchange parity；同步无
   self-edge；step0 equality；六条 edge 与三个 source mixer 梯度；exact Signal；
   B64/K8 8-step capacity；100-step fixed-batch overfit。
2. **Q1 complete-path OOF qualification**：复用 V12 三个 fold 的 Signal/expert
   checkpoints。每折冻结完整 base，只在其94个 source identities 上训练 CRDE
   与 heads 20 epochs，final epoch only；在47个 heldout identities 上评价。总计
   三次小模块训练，不访问30-dev/official。
3. **D1 all-fit main**：仅当 Q1 全门通过，加载 V8 all-fit Phase-A，冻结 base，
   训练同一 CRDE 20 epochs、final epoch only；唯一一次 frozen 30-dev 评价。
4. **After success only**：D1达到65且严格胜出后，才允许 official test、必要
   消融和论文完整 SOTA 表；否则封存 V15。

## Failure Modes and Diagnostics

- **Exchange stays zero**：检查六条 edge 的 `tanh(theta)`、message norm 和梯度；
  Q1 要求 learned edge 至少四条非零，但不以非零本身证明增益。
- **One expert dominates**：Q1 要求 aggregate 三个 receiver 都严格优于其 matched
  no-exchange branch，且每折至少两个 receiver 改善；否则不能声称 mutual
  promotion。
- **Fused improves by damaging branches**：fused 与三个 branch 的 OOF gates 分开；
  `L_coop` 逐输出计算。
- **Fold-specific overfit**：三折 final epoch 都必须 fused 不劣，21-identity
  cluster bootstrap 的 fused gain 95% lower bound 必须大于0。
- **Memory pressure**：base 全冻结，exchange rank64；capacity 实测决定是否能
  用真实 B64/K8。不得用梯度累积冒充 Batch-Hard 的物理 batch。

## Novelty and Elegance Argument

FusionReID 已有 CNN-Transformer 多层交换，MambaPro/PRISM 已有
Transformer-Mamba 或跨模态渐进协同，原 HFER 也做过 full-state low-rank relay。
V15 的可辩护差异不是“三种网络一起用”，而是一个更窄的机制组合：

1. 所有 architecture experts 共享同一 pretrained CLIP tail 语义坐标；
2. 通信对象严格是同层 role adapter 产生的 matched delta，而不是 full feature、
   pooled embedding 或 logits；
3. peer delta 在下一 pretrained tail 之前注入，同时保留 receiver identity path；
4. 每条通信边由严格 no-exchange relational risk counterfactual 监督，而不是
   用局部质量去预测全局检索效用。

这把论文主线收敛为“何时、交换什么、怎样证明交换有益”，而不是并列堆叠
HFER+Router+KD。

## Claim-Driven Validation Sketch

### Claim 1: Stagewise role-delta exchange creates identity-disjoint cooperation

- **Minimal experiment**：V12 三个 complete-path OOF folds，CRDE-on 对严格
  CRDE-off。
- **Metrics**：每折/聚合 mAP、Rank-1、batch-hard risk；21-identity cluster
  bootstrap；三个 receiver 的 matched gain。
- **Decisive gate**：三折 fused gain 均 `>=0`，query-weighted fused mAP gain
  `>=1.0`，bootstrap lower bound `>0`；aggregate CNN/T/M gain 全部 `>0`，每折
  至少两个 receiver `>0`；fused 严格高于三个 exchanged branches。

### Claim 2: The deployable deterministic collaboration clears the main bottleneck

- **Minimal experiment**：通过 Q1 后唯一 all-fit seed42 20-epoch、final-only dev。
- **Compared systems**：同 checkpoint exact Signal、no-exchange V8 bank、CRDE
  fused、CRDE CNN/T/M。
- **Decisive gate**：fused `>=65 mAP`，mAP 严格高于全部 comparator，Rank-1
  严格高于三个 branches；official0。

## Experiment Handoff Inputs

- **Must-prove claims**：exchange 在 full-path identity isolation 下提升 fused 和
  receiver；唯一部署模型达到65并严格胜出。
- **Must-run ablations**：主成功前为零；成功后只做 no-exchange、full-state
  exchange、without `L_coop` 三项最小消融。
- **Critical dataset/metrics**：RGBNT201固定141-fit/30-dev；mAP、Rank-1、
  cross-camera retrieval risk、identity-cluster bootstrap。
- **Highest-risk assumption**：冻结 V8 role experts 上的低秩 delta exchange 是否
  有足够容量产生超出 branch Oracle 的新排序，而不重演 V9 的 late-fusion overfit。

## Compute & Timeline Estimate

- M0：约10–20分钟。
- Q1：复用 V12 checkpoint，不重训 Signal/expert；三折 CRDE 各20 epoch，估计
  30–50分钟总计。
- D1（仅Q1通过）：all-fit 20 epoch约15–25分钟，最终 dev约1分钟。
- 总新增预算：晋级前约1 GPU-hour；通过后累计约1.5 GPU-hours；无标注成本。
