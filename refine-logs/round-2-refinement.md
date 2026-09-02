# Round 2 Refinement

## Problem Anchor

- Bottom-line problem: 在 RGBNT201 的固定 `141-fit / 30-dev` 协议上，把 CNN、Transformer、Mamba 三类异构专家已经被 V12 证明存在的互补空间转化为可部署融合增益，同时严格保留 exact Signal baseline，最终 seed 42 fused mAP 达到至少 65，并严格超过 baseline 与全部固定专家。
- Must-solve bottleneck: V12 的身份 Router 在完整路径 identity-OOF 条件下仍不能跨折泛化；learned expected margin 为 `-0.117330`，低于固定策略 `-0.099975`，Top-slot accuracy 为 `12.2592%`，低于 majority `16.8126%`，而质量响应门已经通过。
- Non-goals: 不复现 baseline；不做多种子；主结果达到 65 mAP 前不做消融；不访问 official test；不以增加另一套大 backbone、HFER 或超参数扫描掩盖 Router 监督失配。
- Constraints: 所有训练、推理和数据处理仅在远端单张 RTX 3090 24 GiB 上进行；固定 seed 42、B64/K8、既有 3-fold identity-OOF 划分和 V8/V12 已登记超参数；复用现有 V12 fold checkpoints，不重新训练 Signal baseline 或 V12 教师。
- Success condition: train-only Q0 证明实际融合路径上的九槽反事实效用非退化且具有专家/模态互补；Q1 learned Router 的 held-out-fold expected counterfactual utility 严格超过固定策略、Top-slot accuracy 严格超过 majority，并继续通过三模态退化响应与 missing-mass=0；只有全部通过才允许一次 final-only dev，且 fused≥65 mAP 并严格超过 exact Signal、V8 Phase-B 和三个固定专家。

## Anchor Check

- Original bottleneck: OOF teacher target、Router 输入和最终融合路径没有闭合。
- Why the revised method still addresses it: target 与 replay 来自 identity-heldout OOF 路径；Router 输入始终来自最终部署的 all-fit feature；同一 fusion 函数用于 target、replay 和 final。
- Reviewer suggestions rejected as drift: 不新增模型或实验族；仅强化同一次 Q0/Q1 的统计门。

## Simplicity Check

- Dominant contribution after revision: Deployment-Aligned Fusion-Path Counterfactual Distillation。
- Components removed or merged: 无 Gram、无 learned alpha、无第二控制器；transfer audit 是 binary prerequisite，quality/Top1 是诊断。
- Reviewer suggestions rejected as unnecessary complexity: 不增加 DINO/HFER/LLM/MoE、新 backbone、参数扫描或主结果前消融。
- Why the remaining mechanism is still the smallest adequate route: 一个 paired cache、一个既有 Router、一个共享 fusion/evaluator seam。

## Changes Made

### 1. 收紧科学 claim 边界

- Reviewer said: all-fit student extractor 已见过全部 141 fit identities，不能声称 representation identity-OOF。
- Action: 明确 Q1 是 fixed all-fit deployment interface 下的 identity-heldout policy validation；只有 target 与 replay evaluation 是完整 identity-OOF。
- Reasoning: 这正好对应 final inference 的输入，不虚构 backbone 未见身份。
- Impact on core method: 支持 policy-transfer claim，不支持 representation-generalization claim。

### 2. 将 transfer audit 隔离

- Reviewer said: all-fit Delta 不能进入训练、阈值或 fixed policy 选择。
- Action: 写死为 one-shot binary prerequisite；训练、fixed policy 和所有阈值均只由协议预注册或 OOF train folds 决定。
- Reasoning: 防止 all-fit fit-label utility 变成 target leakage。
- Impact on core method: all-fit Delta 只回答九槽语义是否完全错位。

### 3. 用 paired identity-cluster inference 替代任意微小阈值

- Reviewer said: `1e-6/1e-4` 不是科学改善。
- Action: per-fold 保留 deterministic non-inferiority；aggregate expected Delta、Top1 和 replay AP 使用 identity-cluster paired bootstrap，固定 seed42/10,000 resamples，95% lower confidence bound 必须 `>0`。replay mAP 成为 hard gate。
- Reasoning: 同一 Q1 内对 query-level paired difference 作 identity-cluster 重采样，尊重同一 identity 内相关性；这不是模型多种子实验。
- Impact on core method: 禁止浮点级偶然改善晋级。

## Revised Proposal

# Research Proposal: TriFusion V13 部署对齐的融合路径反事实蒸馏

## Problem Anchor

- Bottom-line problem: 在 RGBNT201 的固定 `141-fit / 30-dev` 协议上，把 CNN、Transformer、Mamba 三类异构专家已经被 V12 证明存在的互补空间转化为可部署融合增益，同时严格保留 exact Signal baseline，最终 seed 42 fused mAP 达到至少 65，并严格超过 baseline 与全部固定专家。
- Must-solve bottleneck: V12 的身份 Router 在完整路径 identity-OOF 条件下仍不能跨折泛化；learned expected margin 为 `-0.117330`，低于固定策略 `-0.099975`，Top-slot accuracy 为 `12.2592%`，低于 majority `16.8126%`，而质量响应门已经通过。
- Non-goals: 不复现 baseline；不做多种子；主结果达到 65 mAP 前不做消融；不访问 official test；不以增加另一套大 backbone、HFER 或超参数扫描掩盖 Router 监督失配。
- Constraints: 所有训练、推理和数据处理仅在远端单张 RTX 3090 24 GiB 上进行；固定 seed 42、B64/K8、既有 3-fold identity-OOF 划分和 V8/V12 已登记超参数；复用现有 V12 fold checkpoints，不重新训练 Signal baseline 或 V12 教师。
- Success condition: train-only Q0 证明实际融合路径上的九槽反事实效用非退化且具有专家/模态互补；Q1 learned Router 的 held-out-fold expected counterfactual utility 严格超过固定策略、Top-slot accuracy 严格超过 majority，并继续通过三模态退化响应与 missing-mass=0；只有全部通过才允许一次 final-only dev，且 fused≥65 mAP 并严格超过 exact Signal、V8 Phase-B 和三个固定专家。

## Technical Gap

V12 用 residual-only absolute margin 监督最终 `[exact Signal, routed residual bank]`，target 与 action 后果不同；V12 Q1 的 Router 又读取三套 fold raw features，final inference 则读取 all-fit Phase-A feature，student 输入坐标也不同。

V13 用一个 paired cache 闭合两处失配：OOF complete model 生成真实融合 query-side counterfactual target，并提供 non-saturated replay path；同一样本的 frozen all-fit Phase-A feature 是 Router 唯一输入。Router 决策通过 sample key 回放到 OOF path，直接以真实 ReID margin/AP 过门。

## Method Thesis and Contribution

- Thesis: 将 identity-OOF complete-path 的融合贡献效用蒸馏到 fixed deployment feature interface 上的轻量层级 Router，并用 OOF replay 证明该 policy 的真实检索收益。
- Dominant contribution: Deployment-Aligned Fusion-Path Counterfactual Distillation。
- Scientific boundary: Q1 是 fixed all-fit deployment feature interface 下的 identity-heldout policy validation。OOF teacher supervision 与 OOF replay 对 heldout identity 隔离；student feature extractor 本身见过全部 141 fit identities，因此不声称 identity-heldout representation training/generalization。
- Non-contributions: quality gate、hierarchical normalization、exact prefix 和 fixed alpha 是沿用控制，不作 novelty claim。

## Complexity Budget

- Reuse: V12 三个 fold Signal/expert checkpoint；V8 Phase-A all-fit checkpoint；现有 Router；exact Signal evaluator。
- New trainable component: 仍只有既有 Router。
- Fixed simplification: `alpha=0.2`；无 alpha head/loss、Gram、HFER、DINO、额外 backbone 或 joint training。

## Single Shared Fusion Seam

Q0 target、Q1 replay、transfer audit 和 final dev 必须调用同一个公共函数：

`b(x,w)=vec({w_s r_s(x)} in fixed expert-major/modality-minor order)`

`F(x;w)=L2(concat(z0(x),0.2*||z0(x)||_2*L2(b(x,w))))`。

函数在 residual-bank normalization 前应用 weights/mask；pre-final-L2 的前 3072D 必须逐元素等于 exact `z0`。所有有效 clean slots 的 uniform 权重为 `1/9`；remove `s` 时先在 unnormalized bank 置零，再以其余八槽 `1/8` 完整重跑该函数。

## Fixed OOF Bank and Higher-Is-Better Target

每 fold 的 `Q_f` 为具备 different-camera positive 的 heldout fit records（沿用 `190/179/202`）；`G_f` 为相同 ordered records 的复制。同 identity/same camera（包括自身）剔除，positive=`same identity && different camera`，negative=`different identity`。

`B_f={F_f(x_i;w^U):x_i∈G_f}` 固定一次。Euclidean distance 作用于 final L2 embeddings，定义：

`M_f(q,B_f)=min_negative_distance-max_positive_distance`，越大越好。

`Delta_f(q,s)=M_f(F_f(q;w^U),B_f)-M_f(F_f(q;w^{-s}),B_f)`。

因此 `Delta>0` 表示槽 `s` 有益。所有 intervention 都是 query-side；`B_f` 的 tensor SHA 在九次 removal 中必须不变。

## Paired Cache and Provenance

每行按 sample key 存储：

- `teacher_oof_baseline/modal_residual/Delta`；
- `student_allfit_direct/modal_residual`；
- identity/camera/fold/sample path hash。

receipt 绑定 ordered sample hash、identity/camera tensor SHA、fold Signal/expert SHA、all-fit Phase-A SHA、feature tensors SHA、shared fusion/evaluator/config/commit SHA、alpha、slot order 和 mask。final refit 加载的 Phase-A SHA 必须与 `student_*` cache SHA 字段完全一致。

## Read-Only Action-Transfer Prerequisite

同一共享 `F` 可计算 all-fit path Delta，但它严格：

- 不进入 Router loss；
- 不选择 temperature/threshold/fixed slot；
- 不选择 checkpoint 或 epoch；
- 不用于 dev 决策；
- 只执行一次并输出 pass/fail 与 sign agreement、top-slot overlap、Spearman rank correlation 的 report-only 统计。

其唯一 binary gate：由 OOF Delta 选出的 per-query oracle slot，在 all-fit Delta 上的 aggregate utility 必须严格高于仅由 OOF train folds 选出的 fixed slot，且每 fold 不低于 fixed。失败则 action semantics 不转移，Q1 禁止。

## Router Training and OOF Replay

Router 始终读取 `student_allfit_*`，target 为同一行 `teacher_oof_Delta`。保持现有 `P(m|x)P(e|m,x)`、KL、quality KL、100 epochs、LR `3.5e-4`、temperature `0.05`、hidden128、seed42；alpha 常量0.2。

heldout fold 时，Router 对 query/gallery 的 student feature 各自输出 weights；按 sample key 将 weights 应用到 teacher OOF baseline/residual，再调用共享 `F`，按相同 cross-camera filter 产生 per-query margin、AP、Rank-1。

fixed slot 只能由另外两个 OOF training folds 的 mean Delta 选择；majority 也只由 training folds 定义。

## Hard Q0/Q1 Gates

Q0 target health 每 fold报告九槽 Delta mean/std/min/max、positive ratio、unique positive winners、target entropy、full/remove non-identity、bank SHA immutability。聚合后每个 expert/modality 至少有一个 unique positive winner，slot Oracle mean Delta 严格超过 best fixed slot，并通过 read-only action-transfer prerequisite。

Q1 gate：

1. 每 fold learned expected Delta ≥ train-fold fixed expected Delta；
2. 每 fold Top-slot accuracy ≥ train-fold majority accuracy；
3. 每 fold OOF replay mAP ≥ fixed-slot replay mAP，mean margin ≥ fixed-slot replay mean margin；
4. aggregate query-level paired differences分别对 expected Delta、Top1 correctness 和 AP 做 identity-cluster bootstrap：以 identity 为 cluster，固定统计 seed42、10,000 resamples；95% percentile lower bound 必须 `>0`；
5. aggregate replay mean margin 的 identity-cluster bootstrap 95% lower bound也必须 `>0`；
6. RGB/NI/TI corruption 后本模态 mass 均下降，missing mass严格为0；
7. dev/official access均为0，expert/Signal/Phase-A state SHA前后不变。

bootstrap 只重采样已经固定的 query-level paired differences，不重训模型，不属于模型多种子实验，也不能用于调阈值。

任一 hard gate 失败：无 final refit、无 dev、无参数扫描。

## Final Integration and Claim Boundary

Q1 全门通过后，在全部 paired rows 上 final refit Router，并组合与 cache 完全相同 SHA 的 all-fit Phase-A checkpoint。只做一次 30-dev frozen evaluation。fused 必须 ≥65 mAP，并严格超过 exact Signal `58.0109`、当前 deployable V8 Phase-B `58.4050` 和三个 fixed experts；否则不访问 official、不做消融、不声称 SOTA。

若主结果通过，才允许两项 claim-critical deletion checks：`actual-path target→residual-only target`、`paired deployment input→fold raw input`。不增加其他实验族。

## Claim-Driven Validation

### Claim 1: deployment feature 可学习 OOF actual-path utility

- Evidence: paired Q0 + fixed Q1；per-fold non-inferiority；identity-cluster paired lower bound>0；OOF replay mAP/margin hard gain；quality/missing pass。
- Failure meaning: policy 不可迁移；V13 封存。

### Claim 2: policy 可转化为部署主结果

- Evidence: 条件式一次 seed42 dev，fused≥65且严格胜全部登记输出。
- Failure meaning: 仅保留 train-only policy 证据，不授权 official/SOTA。

## Compute & Timeline

- Q0 paired forward/replay 预计 0.1–0.3 GPU-hour；Q1/final Router 数分钟。
- 无教师重训、无 baseline 重跑、无新数据、无多种子。
- 顺序：TDD RED→GREEN → remote preflight → Q0 → 条件式 Q1 → 条件式 dev。
