# 研究方案：TriFusion V16 — Signal 锚定的三方关系修复（SATR）

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

## Technical Gap

这份用户诊断对 V5/V6 是正确的，但其中主要工程修复已经在 V7/V8 完成：

- 三模态共享几何、matched-token residual、层级
  `P(m|x)P(e|m,x)`、有界 `alpha`、missing-modality 零质量、归一化 Triplet、
  label smoothing 和真实 B64/K8 已实现并通过测试；
- V8 Phase-A 的 branch Oracle 为 `64.7850 mAP`，每个专家都有独有 AP 胜例和
  正 leave-one-out 边际，证明专家互补空间真实存在；
- V8 Phase-B 的层级 Router 只把部署结果推到 `58.4050 mAP`；V12 complete-path
  utility、V13 actual deployment-path utility、V14 fold-robust retrieval regret
  均未通过 identity-OOF Router 门；
- V15 取消 Router、改做两阶段 role-delta exchange，但聚合 fused/CNN/T/M
  增益为 `-0.1721/-0.1576/-0.2606/+0.2898` mAP。read-only postmortem 进一步
  排除了“交换太弱”：Transformer 收到相当于自身 role delta `29%–43%` 的注入，
  却仍为负增益。

因此当前缺失的不是另一个质量头，而是一个**不把一个专家的坐标向量强塞进另
一个专家、又能让三者在训练期共享可验证检索知识的接口**。ReID 是 open-set
检索，关系比训练身份 logits 更接近部署目标。CVPR 2019 的 Similarity Knowledge
Distillation 和 RKD 已证明关系蒸馏是已有大类；PMKD 2026 已覆盖 hard-sample
aware progressive multimodal KD；本仓库旧 RDPT 也已有 reliability-selected
pairwise peer teaching。因此 V16 不能声称“关系蒸馏”或“选择教师”本身新颖。

可检验的新缺口更窄：**在三个在线异构对等专家中，用另外两支的交集而非单支
置信度作为教师；只接受同时超过 exact Signal 和当前 receiver 的跨相机排序
关系；并用单向 margin repair 保持各自推理坐标私有。**

## Method Thesis

- **One-sentence thesis**：以 exact Signal 的跨相机排序为不可退化锚点，只有当
  另外两个异构专家对同一正负关系一致、且两者都严格优于 Signal 和 receiver
  时，才把其保守共同 margin 作为 stop-gradient 目标修复 receiver；这种训练期
  三方关系修复可以转移互补判断而不执行推理期特征交换或路由。
- **Why this is the smallest adequate intervention**：只增加一个无参数的
  relation-mask/target 构造器和两项 rank-margin loss；不新增 backbone、Router、
  teacher network、EMA、projection 或 inference module。
- **Why timely in the foundation-model era**：冻结 CLIP/Signal 不再只是 feature
  source，而是稳定、强且逐元素可审计的语义锚点；异构 adapters 只学习修复强
  foundation representation 尚未解决的跨相机关系。

## Contribution Focus

- **Dominant contribution — Signal-Anchored Triadic Repair (SATR)**：一种面向
  open-set ReID 的三方在线关系教学。对每个 receiver，另外两支必须在同一
  cross-camera triplet 上形成共同改进下界，任一 peer 不同意就拒绝教学。
- **Supporting contribution — one-sided anchor protection**：对 Signal 已可靠的
  排序只施加“不可低于锚点”的单向保护，不把 Signal embedding 或完整 similarity
  matrix 强制复制给专家；它与 SATR 共用同一 margin 张量，不是第二个网络模块。
- **Retained foundation design**：V8 的 pretrained-tail semantic rebranching 使
  CNN/Transformer/Mamba 都继承 CLIP 强语义，同时保留 exact Signal prefix；这是
  系统基础，不冒充 V16 新发明。
- **Explicit non-contributions**：不声称首次三架构、首次 mutual KD、首次
  relational KD、首次 hard-sample distillation 或首次 consensus teaching；不把
  fixed fusion、stop-gradient 或 triplet mining 单独包装成创新。

## Proposed Method

### Complexity Budget

- **Frozen/reused**：Signal backbone/exact 3072D output、V8/V12 pretrained-tail
  expert topology、共享几何数据管线、三个 role adapters、现有 BN/classifier 和
  evaluator。
- **Trainable**：三个既有 expert role adapters/heads；不创建新的可训练模块。
- **New computation**：训练期一次 `B x B` cosine matrix/专家，共四个私有输出和
  exact Signal；在有效 cross-camera positive/negative 上构造关系 margin。
- **Intentionally excluded**：Router、reliability posterior、HFER/CRDE、teacher
  checkpoint、EMA、learned alpha、full-matrix MSE、logit KL、private CKA hinge、
  额外预训练模型。

### System Overview

```text
RGB / NIR / TIR --shared geometry--> frozen Signal
        |                                  |
        |                           exact z0 (3072D)
        v
  shared CLIP block8 anchor
        |
        +--> CNN pretrained-tail expert --------> z_C
        +--> Transformer pretrained-tail expert -> z_T
        +--> Mamba pretrained-tail expert ------> z_M
                          |
          training only: normalized cross-camera margins
                          |
      Signal-protected set + two-peer consensus repair set
                          |
             one-sided relation losses update private experts
                          |
 inference: [exact z0, normalized private residual bank]
```

### Core Mechanism：SATR

对 batch 内 query `q`、异 camera 同身份 positive `p` 和异身份 negative `n`，
在 L2-normalized embedding 上定义更大更好的检索 margin：

\[
m_e(q,p,n)=s(z_e^q,z_e^p)-s(z_e^q,z_e^n),
\]

其中 `e in {0,C,T,M}`，`0` 是 exact Signal。所有 teacher margin 和集合 mask
都 `stop-gradient`。

对 receiver `e`，另外两支为 `j,k`，其保守共同知识为：

\[
t_e=\min(m_j,m_k).
\]

只在以下交集上教学：

\[
\mathcal R_e=
\{(q,p,n):
t_e \ge \max(m_0,m_e)+\delta_r,
\quad t_e>0\}.
\]

因为使用 `min`，这等价于两支 peer 都严格胜过 exact Signal 与 receiver；没有
learned confidence，也没有从单支的偶然高分推断教师。receiver 只补足缺口：

\[
L_{repair}^e=
\frac{1}{|\mathcal R_e|}
\sum_{\mathcal R_e}
\operatorname{smoothL1}
\left(\max(0,t_e-m_e),0\right).
\]

当 receiver 已达到保守共同 margin 时损失为零；它不要求 receiver 复制 peer
embedding、绝对距离或 identity logits。

同一个 batch 中 CNN、Transformer、Mamba 都轮流充当 receiver，因此互教是
对称角色合同，但每个 triplet 的实际教学方向可以不同或全部拒绝。SATR 不需要
将某支永久定义为强教师。

### Supporting Component：Signal 单向关系保护

对 exact Signal 已正确且有余量的关系：

\[
\mathcal P=\{(q,p,n):m_0\ge\gamma_p\},
\]

只约束最终 fused 不跌破 Signal 的容忍下界：

\[
L_{protect}=
\frac1{|\mathcal P|}\sum_{\mathcal P}
\max(0,m_0-\epsilon_p-m_{fused})^2.
\]

保护只作用于 fused，避免强迫三个私有专家同质化。SATR 修复 Signal/receiver
尚未处理好的关系，`L_protect` 保住强锚点已经解决的关系，两者形成“保留 +
修复”的单一训练逻辑。

### Training Objective

保留 V8 Phase-A 的 expert-specific ID、L2-normalized triplet 和 fused ReID loss：

\[
L=L_{reid}^{fused}+
\lambda_b\sum_{e\in\{C,T,M\}}L_{reid}^e+
\lambda_r\sum_e L_{repair}^e+
\lambda_pL_{protect}.
\]

首版只注册一组固定系数，不做扫描。若某 batch 某 receiver 的 repair set 为空，
该项按数学定义贡献 0；这是选择机制的正常拒绝状态，不增加 fallback teacher。

### Inference Path

推理时删除全部 relation 构造和 loss。不存在 peer call、Router 或 exchange：

\[
z_{fused}=[z_0,\|z_0\|\operatorname{Normalize}
(\operatorname{Concat}(r_C,r_T,r_M))].
\]

输出仍包括 exact `baseline_only`、fused、CNN、Transformer、Mamba。部署 FLOPs
与无 SATR 的 V8 expert bank 相同，三个专家的 embedding 坐标保持私有。

### 与旧 RDPT 的边界

| 维度 | RDPT 草案 | V16 SATR |
|---|---|---|
| 教师选择 | reliability gap 的单 peer 方向 | 两个 peer 的 relation intersection |
| 教学单位 | sample-level teacher；logit/role payload | cross-camera `(q,p,n)` margin |
| 锚点 | URGC reliability | exact Signal relation |
| 传输内容 | KL + adapted role relation | 仅保守共同排序 margin |
| 拒绝条件 | 教师 quality 不足 | 任一 peer 未同时胜过 Signal/receiver |
| 推理表示 | private hinge 保护 | 从不匹配 embedding，天然私有 |

因此 SATR 不是把 RDPT 改名：它删除 reliability、KL、payload adapter 和 private
hinge，把监督对象改为由 exact foundation anchor 验证的三方关系交集。

## Training Plan

1. **M0 public seam / engineering qualification**：真实 RED→GREEN；验证 margin
   符号、cross-camera mask、两-peer intersection、任一 peer 不同意则零教学、
   teacher stop-gradient、三 receiver 梯度、空集合为精确零、Signal exact parity、
   B64/K8 8-step capacity、100-step fixed-batch overfit。
2. **Q1 complete-path identity-OOF**：复用 V12 三个 fold checkpoint；每折在其
   94 source identities 上从固定 checkpoint 训练同一组三 expert adapters/heads
   20 epochs，final epoch only；在 47 unseen identities 上与同 checkpoint、同
   schedule、`lambda_r=lambda_p=0` 的 matched comparator 比较。为了不额外训练
   一套 comparator，no-SATR 端点使用冻结的进入 V16 前 checkpoint；训练步数和
   optimizer scope必须在计划阶段明确绑定，审查其公平性。
3. **D1 all-fit main（仅 Q1 全门通过）**：从 V8 all-fit Phase-A checkpoint
   启动同一 SATR 配方，20 epochs、final checkpoint only；训练期间 dev0，结束后
   唯一 frozen 30-dev 评估。
4. **终止规则**：Q1 失败即封存 V16；不访问 dev。D1 未达到65即封存；不进入
   official、消融或多 seed。

## Failure Modes and Diagnostics

- **共识集合饱和**：记录每个 receiver 的 eligible triplet 比例和三者交集；若
  接近全量，说明门没有选择性，Q1 不晋级。
- **共识集合为空**：记录每个 identity/receiver 的非零覆盖；若某专家在全部
  source/heldout 上始终无关系可学，不能声称三方互促。
- **共同错误被强化**：`t_e>0` 且严格超过 exact Signal 的条件使用训练 GT 排序
  验证，不把 peer agreement 本身解释为正确。
- **专家同质化**：SATR 不匹配 embedding 或全 similarity matrix；仍记录三支
  residual-only 的 unique AP wins 和 pairwise CKA 作为诊断，但主成功前不做
  超参修复。
- **只改善 fused、伤害分支**：Q1 为 CNN/T/M 分别设 matched gain 门；fused
  不得靠牺牲某一支通过“互促”主张。

## Novelty and Elegance Argument

已有工作分别覆盖 similarity/RKD、multi-teacher weighting、mutual ReID
distillation、hardness-aware progressive multimodal KD 和 heterogeneous KD。
因此可辩护差异不是这些大类，而是以下不可拆的机制组合：

1. exact Signal 作为每个训练 triplet 的 foundation relation anchor，而非可学习
   reliability 或 teacher score；
2. 三专家中的 receiver 只接收另外两支共同认可的 improvement lower bound，
   不是平均 ensemble 或 best-teacher selection；
3. 一侧 margin repair 传递“哪条排序需要怎样修复”，不传坐标向量、logits 或
   token，并在推理时完全移除协同接口。

这直接回应 V15 的因果证据：既然 hidden direction 在身份折间反转，就只转移
坐标无关、经两支与强锚点共同验证的 retrieval relation。方法只有一个主机制和
一个共用 margin 张量的保护项，避免再次堆叠 Router、exchange 与 KD。

## Claim-Driven Validation Sketch

### Claim 1：SATR 在 identity-disjoint 路径上产生稳定三方关系互促

- **Minimal experiment**：V12 三个 complete-path folds；SATR endpoint 对同起点
  no-SATR comparator。
- **Metrics**：每折/聚合 fused、CNN、Transformer、Mamba mAP/Rank-1；
  21-identity cluster bootstrap；repair coverage/rejection；三支 unique wins。
- **Decisive gate**：三折 fused gain 均 `>=0`；query-weighted fused mAP gain
  `>=+1.0`；bootstrap lower bound `>0`；aggregate CNN/T/M gain 全部 `>0`，每折
  至少两支 `>0`；repair coverage 非零且非饱和。

### Claim 2：训练期关系协同转化为无需 Router/exchange 的部署增益

- **Minimal experiment**：Q1 通过后唯一 seed42 all-fit D1、final-only dev。
- **Comparators**：exact Signal、当前 V8 Phase-B deployable `58.4050`、V16
  CNN/T/M。
- **Decisive gate**：fused `>=65 mAP`；mAP 严格高于全部 comparator；Rank-1
  严格高于三分支；official access=0。

## Experiment Handoff Inputs

- **Must-prove claims**：两-peer/Signal 验证的关系修复能跨身份迁移，并让三个
  私有专家及 fixed fused 共同改善。
- **Must-run ablations**：主成功前为零；成功后最小三项为 no relation loss、
  single-best-peer 替代 two-peer intersection、去掉 Signal protect。
- **Critical dataset/metrics**：RGBNT201 141-fit/30-dev；mAP、Rank-1、
  identity-cluster bootstrap、eligible/rejection rate、unique AP wins。
- **Highest-risk assumption**：batch 内 B64/K8 的两-peer一致关系是否有足够覆盖，
  且这种在线选择是否在 unseen identities 上比静态 hidden exchange 更稳定。

## Compute & Timeline Estimate

- M0：约 15–25 分钟。
- Q1：三折各20 epoch；估计 35–60 分钟总计。
- D1（仅Q1通过）：约20–30分钟训练 + 1分钟最终 dev。
- 新增预算：晋级前约1 GPU-hour；通过后累计不超过1.5 GPU-hours；无新增标注。

## Grounding Sources

- [Distilled Person Re-Identification: Towards a More Scalable System, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Distilled_Person_Re-Identification_Towards_a_More_Scalable_System_CVPR_2019_paper.html)
- [Relational Knowledge Distillation, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html)
- [Meta Pairwise Relationship Distillation for Unsupervised Person Re-ID, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Ji_Meta_Pairwise_Relationship_Distillation_for_Unsupervised_Person_Re-Identification_ICCV_2021_paper.html)
- [Progressive Multi-modal Knowledge Distillation for Multi-spectral Object Re-ID, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38338)
- [Heterogeneous Complementary Distillation, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38112)

