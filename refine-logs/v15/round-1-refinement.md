# Round 1 Refinement

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

## Anchor Check

- **Original bottleneck**：fixed experts 的选择上限不足、Router 无法泛化、late
  pooled relay 有害，缺少能在 pretrained semantic path 内创造新表示的机制。
- **Why preserved**：修订仍只在 V8 tail stage 内交换 CNN/T/M role delta，并以
  no-exchange retrieval risk 约束；没有改成更容易的分类、质量预测或 Oracle。
- **Drift rejected**：不把 V12 OOF 高指标当部署主结果，也不引入额外 backbone、
  LLM/VLM、更多路由或阈值扫描。

## Simplicity Check

- **Sole contribution**：delta-only pre-tail expert exchange trained by matched
  no-exchange retrieval regret。
- **Removed**：删除第三层 exchange；V8 semantic rebranching 只作为冻结底座；
  matched regret 不再作为第二贡献；删除“至少四条非零边”科学门。
- **Frozen**：Signal、V8 role experts、CLIP tail、no-exchange comparator 全冻结且
  eval；只训练两层 CRDE 与 source-local on-path BN/classifier heads。

## Reviewer Feedback Actions

1. **Off counterfactual state mutation — accepted / CRITICAL**：off 由完全冻结、
   `eval()` 的 V8 base 在 `torch.no_grad()` 下产生 pre-BN embedding；不调用 V15
   BN/classifier；`R_off` 显式 detach。
2. **BN/classifier ownership — accepted**：每个 fold/all-fit run 新建只覆盖其 source
   class labels 的 on-path BNNeck/classifier；heldout evaluation 不读取 logits，所有
   retrieval 使用 pre-BN embeddings。
3. **Stage-3 ambiguity — accepted**：只在 tail9、tail10 的 role adapter 后交换，
   分别由 frozen tail10、tail11 解释；tail11 后不交换。
4. **Contribution sprawl — accepted**：V8 是 substrate，不列贡献；CRDE 与其
   matched-regret training 是一个不可分机制。
5. **OOF scope — accepted**：Q1 只授权 D1，不支持65或部署 claim。

## Revised Proposal

### Technical Gap

V8 在 CLIP block8 后将同一 pretrained semantic field 分成 CNN local、
Transformer global 和 Mamba spatial/cross-modal 三条路径。三支有真实 unique
wins，但只在最终 residual bank 相遇；hard selection Oracle 仍低于65。V13/V14
进一步否定用 sample-local Router 恢复 query-gallery relational utility；V9 在
池化后交换完整 residual，强制正交并追加 synthesis，导致 beta 饱和和 dev 降低。

因此最小缺口是：**只把 source expert 在当前层新增的 role information 传给
peers，并让后续 frozen CLIP tail 将其重新解释；通信是否有益直接由 matched
no-exchange retrieval risk 判断。**

### Method Thesis and Contribution

唯一贡献是 **Counterfactual Role-Delta Exchange (CRDE)**：在 tail9/tail10 的
role adapter 后，三专家同步交换 delta-only low-rank messages，保留 receiver
private identity path；训练时以完全冻结、无状态变化的 no-exchange pre-BN
retrieval embedding 为 matched counterfactual，直接最小化通信引起的相对检索
风险。

V8 pretrained-tail rebranching、Signal exact prefix、role adapters/heads、fusion
与评估器全部是复用底座，不作为 V15 新贡献。没有 Router、learned alpha/beta、
末端 synergy head、orthogonalization、DINO、文本或额外 backbone。

### Architecture and Data Flow

```text
shared-geometry RGB/NI/TI
          |
 frozen Signal -> exact baseline z0
          |
 CLIP block8 anchor
          |
 frozen tail9 -> C/T/M role adapters -> role delta D1
          |                              |
          |                         CRDE exchange 1
          |                              |
 frozen tail10 <- receiver private state + bounded peer delta
          |
 C/T/M role adapters -> role delta D2 -> CRDE exchange 2
          |
 frozen tail11 <- receiver private state + bounded peer delta
          |
 matched final residual heads -> C/T/M residual bank
          |
 [exact z0, equal-energy normalized bank]
```

CRDE 只有两个 stages。stage `l` 保存 role adapter 输入 `X_s^l` 与输出
`Y_s^l`，通信源严格为：

\[
D_s^l=Y_s^l-X_s^l.
\]

source-role mixer 在 rank64 空间工作：CNN 对 patch delta 做 depthwise 2D local
mixing；Transformer 对 CLS+patch delta 做 self-attention；Mamba 对 patch delta
复用 four-direction spatial 和 aligned modal scan。六条 `s→t, s≠t` edge 有独立
output projection，同步更新 receiver：

\[
\widetilde Y_t^l=Y_t^l+
\sum_{s\ne t}0.25\tanh(\theta_{s\to t}^l)P_{s\to t}M_s(D_s^l).
\]

`theta=0` 初始化，所以 step0 on/off 完全相等。CRDE 不对 receiver full state
做新 LayerNorm，不正交化、不重标定 message；固定上界0.25不扫描。更新分别在
进入 frozen tail10/tail11 前完成。tail11 后不再交换。

### Exact Trainable/Frozen Boundary

每次 fold 或 all-fit run：

- 加载并冻结对应 V12 fold-specific 或 V8 all-fit `Signal + expert` 完整状态；
- frozen base 始终 `eval()`，参数 `requires_grad=False`，但 on-path 允许梯度穿过
  frozen operations 回到更早 CRDE；
- 新建该 run 专属 CRDE、四个 BNNeck 和四个 classifier；classifier 类别数只等于
  source training identities；
- 只训练 CRDE、on-path BNNeck/classifier；不加载或更新旧 classifier/BN；
- heldout/evaluation 完全绕过 BNNeck/classifier，输出 pre-BN embedding。

### Matched Counterfactual Semantics

每个 source batch 先在 `torch.no_grad()` 中调用冻结 V8 base 的原始
no-exchange forward，直接取得 baseline/fused/CNN/T/M 的 **pre-BN retrieval
embeddings**。base 为 `eval()` 且无 V15 head，因此这条 path 无 running-stat、
dropout、classifier 或共享状态更新；所有 off tensors 立即 detach。

随后只运行一次 CRDE-on forward。它共享相同输入与冻结权重，但执行两个交换
stage；训练 ID loss 只使用 on-path BN/classifier。off 不经过任何 V15 head。

对四个 pre-BN output `o∈{fused,C,T,M}`，先 L2 normalize，使用当前物理
B64/K8 中 cross-camera hardest-positive / nearest-negative：

\[
R(z)=|Q|^{-1}\sum_q
\operatorname{softplus}(d^+_{hard}(q)-d^-_{near}(q)).
\]

若当前 batch 某 query 没有 cross-camera positive，则它不进入 `Q`；训练 loader
和 K8 保证 batch 中存在有效 query，M0 断言 `|Q|>0`。通信 regret 与 V8 既有
ID/triplet 共同组成一个训练目标：

\[
L=L_{V8,on}+
\frac14\sum_o
\operatorname{softplus}(R(z_o^{on})-\operatorname{stopgrad}R(z_o^{off})).
\]

不使用 temperature、margin scan、per-query action target 或 heldout feedback。
softplus 在 step0 risk 相等时仍提供梯度。

### Deterministic Inference

推理只运行一次 CRDE-on 且绕过所有 classifier。正式 fused 保持确定性 V8
表示形式：

\[
z_{fused}=[z_0,\|z_0\|\operatorname{Normalize}
(\operatorname{Concat}(r_C,r_T,r_M))].
\]

同时输出 exact `baseline_only`、CRDE `CNN/Transformer/Mamba`。没有 sample
Router、alpha 或 beta；baseline prefix 必须逐元素等于 Signal。

### Validation and Execution Order

#### M0 — engineering seams

- synthetic role-delta construction；两层、同步、无 self-edge；step0 on/off exact；
- off path `no_grad`、无 V15 BN/classifier 调用、重复 forward state SHA 不变；
- heldout evaluation 只读 pre-BN；exact Signal prefix；
- 六条 directed edges/source mixers 经过更新后都有 finite gradient（非零边数量
  只记录，不门控）；
- remote real B64/K8 8-step capacity 和100-step fixed-batch overfit。

#### Q1 — complete-path identity-OOF qualification

复用 V12 三个 fold checkpoint。每折 base 只见过94 source identities；新 CRDE
和 source-local heads 也只在同94 identities 训练20 epochs，final epoch only；在
其47 heldout identities 上只评价 pre-BN retrieval。各 fold 的 no-exchange
comparator 就是加载前同一 V12 frozen checkpoint，禁止读取30-dev/official。

Q1 只支持“在完整路径身份隔离下，CRDE 相对 matched no-exchange 有可泛化
收益”，只可授权 D1。它不支持65、部署或 SOTA。

预注册 gate：

- 三折 CRDE fused mAP gain 均 `>=0`；
- 571 query 加权 fused mAP gain `>=1.0`；
- 21 identity cluster bootstrap fused gain 95% lower bound `>0`；
- aggregate CNN/T/M matched mAP gain 各 `>0`，每折至少两个 receiver gain `>0`；
- aggregate CRDE fused 严格高于三个 CRDE branches；
- exact prefix/state/access/finite/physical-B64 contracts 全通过。

#### D1 — sole deployable main

只有 Q1 全过，才加载 V8 all-fit Phase-A，冻结 base，以相同 seed42、B64/K8、
loss 和20 final-only epochs 训练 CRDE；随后 frozen checkpoint 对30-dev只评一次。

D1 gate：fused `>=65 mAP`；mAP 严格高于 exact Signal、frozen no-exchange V8
bank 和 CRDE C/T/M；Rank-1 严格高于三个 CRDE branches；official0。失败即封存
V15，不扫参数、不消融、不多seed。

### Failure Modes and Claim Boundary

- Q1 若失败，说明该固定 CRDE 不能把 full-path OOF complementarity 转为稳定
  exchange benefit；不进入 D1。
- Q1 通过而 D1 失败，只能声称 train-only mechanism qualification，不能声称
  deployable gain；V12与V8坐标差异是 scope limitation，不作因果借口。
- edge 参数非零、训练 loss 下降、显存合格都不是科学增益证据。
- 只有 D1 达65且严格胜出后，才允许 no-exchange/full-state/without-regret 三项
  最小消融和 official test；SOTA 仍必须基于相同 official protocol 的真实比较。

### Novelty and Elegance

FusionReID 的多层 CNN-Transformer transfer、MambaPro/PRISM 的渐进协同和原
HFER 的 full-state relay 都阻止宽泛“深层异构交换”新颖性。CRDE 的窄区别是：
**通信内容限定为 matched role-adapter delta；通信位置限定为下一 frozen CLIP
tail 之前；receiver full state 保留恒等路径；训练目标限定为无状态、pre-BN、
matched no-exchange retrieval regret。** 这也是它区别于 V9 late pooled
orthogonal relay 和 V13/V14 routing 的完整机制。

### Compute Estimate

- M0：10–20分钟；
- Q1：三折只训练 CRDE/heads，预计30–50分钟；
- D1（条件执行）：20 epoch约15–25分钟，dev约1分钟；
- 晋级前约1 GPU-hour，总计约1.5 GPU-hours；单张3090，无新数据/标注。
