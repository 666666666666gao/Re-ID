# Round 2 Refinement

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

- 原问题仍是生成超出选择/路由上限的新联合表示；没有换成质量预测或更容易的
  fit 指标。
- V12 OOF 只作为 fail-closed authorization，D1 是唯一部署主结果。
- 审查者因尚无结果而保留 venue risk 是合理风险陈述，但 refinement 的 READY
  只表示方案具体到可以执行，不预判 Q1/D1 会成功；结果门继续严格关闭。

## Simplicity Check

- 唯一贡献：**delta-only pre-tail expert exchange trained by a matched,
  state-clean no-exchange retrieval regret**。
- 只有两层 CRDE；无 Router、alpha/beta、stage3 exchange、末端 synthesis、外部
  backbone 或额外 teacher。
- `lambda_regret=1.0` 是唯一冻结系数，写入 config/provenance/test，不扫描。

## Round-2 Feedback Actions

1. 将 `REGRET_WEIGHT: 1.0` 明确登记为不可覆盖的 V15 contract，并在 builder、
   criterion、config、receipt 四处一致检查。
2. runner 对每个 loader batch 只取一次 `images` mapping；off/on 接收同一组 tensor
   object，M0 以 data pointer equality 断言。禁止在两条 path 间再次调用 transform。
3. M0 新增 off-path hook：V15 BN/classifier forward count 必须为0；全部 frozen
   parameter/buffer SHA 和 BN running mean/var 前后相同。
4. Q1 仍只授权 D1；不因 reviewer 认为方案可行就降低任何科学门。

## Revised Proposal

### Technical Thesis

V8 的三专家各自通过 frozen CLIP tail9/10/11 并有互补 unique wins，但最终才
相遇；selection Oracle 低于65，Router OOF transfer 失败，V9 late pooled relay
有害。V15 只交换同层 role adapter 相对输入新增的 delta，并在下一 frozen CLIP
tail 前注入 peers；matched no-exchange risk 直接约束通信不得恶化检索关系。

### Sole Mechanism: Counterfactual Role-Delta Exchange (CRDE)

V8/Signal 是冻结 substrate。对 tail9/tail10 后的 stage `l` 与 source `s`：

\[
D_s^l=Y_s^l-X_s^l,
\qquad
\widetilde Y_t^l=Y_t^l+
\sum_{s\ne t}0.25\tanh(\theta_{s\to t}^l)P_{s\to t}M_s(D_s^l).
\]

`X/Y` 是 role adapter 前/后 token。CNN mixer 在 rank64 patch grid 做 depthwise
2D local mixing；Transformer mixer 对 rank64 CLS+patch delta 做 self-attention；
Mamba mixer 对 rank64 patch delta 做 four-direction spatial 与 aligned modal
scan。六条 directed peer edges 独立投影、同步更新、无 self-edge。`theta=0`
初始化，绝对通信系数固定不超过0.25。receiver full state 不做额外归一化、
正交化或能量匹配。

CRDE 仅有两层：tail9 后的消息由 frozen tail10 解释，tail10 后的消息由 frozen
tail11 解释；tail11 后不交换。最终沿 V8 matched reference/head 得到 C/T/M
residual bank，正式输出为：

\[
z_{fused}=[z_0,\|z_0\|\operatorname{Normalize}
(\operatorname{Concat}(r_C,r_T,r_M))].
\]

`z0` 必须逐元素等于 exact Signal。推理无 Router、sample scalar 或 classifier。

### Exact State and Head Ownership

每个 Q1 fold 或 D1 run 加载对应 V12/V8 base，随后：

- Signal、CLIP tails、V8 role adapters/heads 和旧 BN/classifier 全冻结并 `eval()`；
- 只训练两层 CRDE，以及该 run 新建的 on-path BNNeck/classifier；
- 新 classifier 类别数只等于 source training identity 数；
- heldout/dev 评价绕过所有 BN/classifier，使用 pre-BN retrieval embeddings；
- on-path 可穿过 frozen ops 向 CRDE 反传，但 frozen parameter/buffer 不更新。

### Matched, State-Clean Counterfactual

loader 每个 batch 只执行一次共享几何/质量 transform，得到一个 `images` mapping。
runner 将**相同 tensor objects**依次传给两条 path：

1. `off`：原 frozen V8 base，`eval()` + `torch.no_grad()`，直接返回 detached
   pre-BN baseline/fused/C/T/M；不实例化或调用 V15 head；
2. `on`：相同 batch tensors，通过两层 CRDE；ID loss 只调用 on-path 的新
   BN/classifier。

off 不更新 running stats/dropout/classifier/shared state。其所有输出与 risk 均
stop-gradient。

对四个输出 `o∈{fused,C,T,M}` 的 L2-normalized pre-BN embedding，计算当前
物理 B64/K8 内 cross-camera hardest-positive / nearest-negative softplus risk：

\[
R(z)=|Q|^{-1}\sum_{q\in Q}
\operatorname{softplus}(d^+_{hard}(q)-d^-_{near}(q)).
\]

总损失冻结为：

\[
L=L_{V8,on}+\lambda_{regret}\frac14\sum_o
\operatorname{softplus}(R(z_o^{on})-
\operatorname{stopgrad}R(z_o^{off})),
\quad \boxed{\lambda_{regret}=1.0}.
\]

`REGRET_WEIGHT` 在 config 必须精确为1.0；builder/criterion/runner/result receipt
不一致即失败。没有 temperature、margin、edge scale 或 loss-weight 扫描。

### M0 Engineering Contract

- role delta、两层同步无 self-edge、step0 on/off exact；
- off/on image keys、shape、storage data pointer 逐 tensor 相同；
- off forward 期间 V15 BN/classifier hooks 计数严格为0；
- frozen parameter+buffer SHA、全部 BN running mean/var/count 前后相同；
- heldout evaluator不调用 classifier，pre-BN only；
- exact Signal prefix、finite gradients、两次 exchange live path；
- `REGRET_WEIGHT==1.0` 四处一致；
- remote real B64/K8 8-step capacity、100-step fixed-batch overfit。

### Q1 Complete-Path OOF Authorization

复用 V12 三个 checkpoint；各 base 的 Signal/expert 与 CRDE训练都只见94 source
identities，训练20 final-only epochs，在47 heldout identities（合计21个
cross-camera eligible identities/571 queries）评价 pre-BN retrieval。记录每折
fit/heldout IDs、Signal/expert SHA、source SHA、access count。no-exchange 是同一
fold原始 frozen V8 checkpoint。

全部门通过才授权 D1：

- 三折 fused mAP gain 均 `>=0`；
- 571-query weighted fused mAP gain `>=1.0`；
- 21-identity cluster bootstrap 95% lower bound `>0`；
- aggregate C/T/M matched mAP gain 各 `>0`，每折至少两个 receiver `>0`；
- aggregate fused 严格高于三个 on branches；
- M0/state/access/fold isolation 全通过。

Q1 不支持65、部署、official或SOTA；失败即封存 V15。

### D1 Sole Main Result

仅Q1通过后，加载 V8 all-fit Phase-A，冻结 base，以相同 seed42、B64/K8、
CRDE、`lambda_regret=1.0` 和20 final-only epochs训练；随后 frozen checkpoint 对
30-dev只评价一次。

晋级要求：fused `>=65 mAP`；mAP 严格高于 exact Signal、frozen no-exchange
V8 bank、CRDE C/T/M；Rank-1严格高于三 branches；official0。失败不扫参、
不消融、不多seed。成功后才允许 no-exchange/full-state/without-regret 三项
论文最小消融与一次 official protocol 评估；同协议真实超过公开最佳前不称SOTA。

### Novelty and Scope

不能声称多分支、深层 exchange 或 CNN+Transformer+Mamba 组合首次。窄机制
区别是：通信内容只含 matched role-adapter delta；位置只在下一 frozen CLIP
tail 前；receiver private state 是恒等主路径；训练比较同一增强 tensor 上无状态
pre-BN no-exchange relational risk。它同时区别于原 HFER full-state relay、
FusionReID transfer、V9 pooled orthogonal synthesis 和 V13/V14 Router。

### Compute

M0 10–20分钟；Q1三折约30–50分钟；条件D1约15–25分钟+1分钟dev。晋级前
约1 GPU-hour，总计约1.5 GPU-hours；单张3090，无新数据或标注。
