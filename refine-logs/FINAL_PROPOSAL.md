# Research Proposal: TriFusion V13 部署对齐的融合路径反事实蒸馏

## Problem Anchor

- Bottom-line problem: 在 RGBNT201 的固定 `141-fit / 30-dev` 协议上，把 CNN、Transformer、Mamba 三类异构专家已经被 V12 证明存在的互补空间转化为可部署融合增益，同时严格保留 exact Signal baseline，最终 seed 42 fused mAP 达到至少 65，并严格超过 baseline 与全部固定专家。
- Must-solve bottleneck: V12 的身份 Router 在完整路径 identity-OOF 条件下仍不能跨折泛化；learned expected margin 为 `-0.117330`，低于固定策略 `-0.099975`，Top-slot accuracy 为 `12.2592%`，低于 majority `16.8126%`，而质量响应门已经通过。
- Non-goals: 不复现 baseline；不做多种子；主结果达到 65 mAP 前不做消融；不访问 official test；不增加大 backbone、HFER 或超参数扫描来掩盖 Router 监督失配。
- Constraints: 所有训练、推理和数据处理仅在远端单张 RTX 3090 24 GiB 上进行；固定 seed 42、B64/K8、既有 3-fold identity-OOF 划分和 V8/V12 已登记超参数；复用 V12 fold checkpoints，不重新训练 Signal baseline 或 V12 教师。
- Success condition: Q0 实际融合路径九槽效用非退化且可转移；Q1 的 expected utility、OOF replay AP/margin、质量与 missing gates 全通过；此后唯一一次 dev 的 fused≥65 mAP，并严格超过 exact Signal、V8 Phase-B 与全部固定专家。

## Technical Gap

V12 用 residual-only absolute margin 监督最终 `[exact Signal, routed residual bank]`，target 与 action 后果不同；V12 Q1 的 Router 又读取三套 fold raw features，final inference 则读取 all-fit Phase-A feature，student 输入坐标也不同。

V13 用一个 paired cache 闭合两处失配：OOF complete model 生成真实融合 query-side counterfactual target，并提供 non-saturated replay path；同一样本的 frozen all-fit Phase-A feature 是 Router 唯一输入。Router 决策通过 sample key 回放到 OOF path，直接以真实 ReID margin/AP 过门。

## Method Thesis and Claim Boundary

- Thesis: 将 identity-OOF complete-path 的融合贡献效用蒸馏到 fixed deployment feature interface 上的轻量层级 Router，并用 OOF replay 证明 policy 的真实检索收益。
- Dominant contribution: Deployment-Aligned Fusion-Path Counterfactual Distillation。
- Scientific boundary: Q1 是 fixed all-fit deployment feature interface 下的 identity-heldout policy validation。OOF teacher supervision 与 OOF replay 对 heldout identity 隔离；student feature extractor 本身见过全部 141 fit identities，因此不声称 identity-heldout representation training/generalization。
- Non-contributions: quality gate、hierarchical normalization、exact prefix 和 fixed alpha 是沿用控制，不作 novelty claim。

## Complexity Budget

- Reuse: V12 三个 fold Signal/expert checkpoints；V8 Phase-A all-fit checkpoint；现有 Router；exact Signal evaluator。
- New trainable component: 仍只有既有 Router。
- Fixed simplification: `alpha=0.2`；无 alpha head/loss、Gram、HFER、DINO、额外 backbone、joint training 或超参数扫描。

## Single Shared Fusion Seam

Q0 target、Q1 replay、transfer audit 和 final dev 必须调用同一个公共函数：

`b(x,w)=vec({w_s r_s(x)} in fixed expert-major/modality-minor order)`

`F(x;w)=L2(concat(z0(x),0.2*||z0(x)||_2*L2(b(x,w))))`。

函数在 residual-bank normalization 前应用 weights/mask；pre-final-L2 的前 3072D 必须逐元素等于 exact `z0`。clean uniform 权重为 `1/9`；remove `s` 时先在 unnormalized bank 置零，再以其余八槽 `1/8` 完整重跑同一函数。

## Fixed OOF Bank and Counterfactual Target

每 fold 的 `Q_f` 为具备 different-camera positive 的 heldout fit records（沿用 `190/179/202`）；`G_f` 为相同 ordered records 的复制。同 identity/same camera（包括自身）剔除，positive=`same identity && different camera`，negative=`different identity`。

`B_f={F_f(x_i;w^U):x_i∈G_f}` 固定一次。Euclidean distance 作用于 final L2 embeddings，定义 higher-is-better margin：

`M_f(q,B_f)=min_negative_distance-max_positive_distance`。

`Delta_f(q,s)=M_f(F_f(q;w^U),B_f)-M_f(F_f(q;w^{-s}),B_f)`。

因此 `Delta>0` 表示槽 `s` 有益。intervention 只发生在 query side；`B_f` 的 tensor SHA 在九次 removal 中必须不变。

## Paired Teacher–Student Cache

每行按 sample key 存储 `teacher_oof_baseline/modal_residual/Delta`、`student_allfit_direct/modal_residual`、identity/camera/fold/sample path hash。receipt 绑定 ordered sample、checkpoints、feature tensors、shared fusion/evaluator/config/commit、alpha、slot order 和 mask 的 SHA。final refit 的 Phase-A SHA 必须与 cache 完全一致。

## Read-Only Action-Transfer Prerequisite

all-fit path Delta 只执行一次，不进入 Router loss，不选择 fixed policy、temperature、threshold、checkpoint/epoch 或 dev 决策。sign agreement、top-slot overlap 与 Spearman 只报告。唯一 binary gate：OOF Delta 的 per-query oracle slot 在 all-fit Delta 上 aggregate 严格优于仅由 OOF train folds 选出的 fixed slot，且每 fold不低于 fixed。

## Router Training and OOF Replay

Router 始终读取 `student_allfit_*`，target 为同一行 `teacher_oof_Delta`。保留 `P(m|x)P(e|m,x)`、utility KL、quality KL、100 epochs、LR `3.5e-4`、temperature `0.05`、hidden128、seed42；alpha 常量0.2。

heldout fold 时，Router 对 query/gallery student features 分别输出 weights；按 sample key 将 weights 应用到 teacher OOF baseline/residual，再调用共享 `F`，按相同 cross-camera filter 产生 per-query margin、AP、Rank-1。fixed slot 与 majority 只能由另外两个 OOF training folds定义。

## Hard Gates

Q0 需要通过 target health、bank immutability、三专家/三模态 unique positive、slot Oracle 和 action-transfer prerequisite。

Q1 必须同时满足：每 fold expected Delta、Top1、OOF replay mAP 和 mean margin 均不低于 train-fold fixed/majority；aggregate query-level paired difference 对四项指标按 identity cluster（保留 cluster 全部 queries）以固定统计 seed42 做10,000次 bootstrap，95% percentile lower bound均 `>0`；三模态 corruption mass下降、missing mass为0；所有冻结 state SHA不变；dev/official access为0。

bootstrap 不重训模型，不属于模型多种子，且不能用于调参。任一失败即停止。

## Final Integration

Q1 全门通过后才在全部 paired rows 上 final refit，并组合相同 SHA 的 Phase-A checkpoint；只做一次 30-dev。fused 必须 ≥65 mAP，并严格超过 exact Signal `58.0109`、V8 Phase-B `58.4050` 和三个 fixed experts。失败则不访问 official、不做消融、不声称 SOTA。

主结果通过后只允许两项 claim-critical deletion checks：`actual-path target→residual-only target`、`paired deployment input→fold raw input`。

## Compute and Run Order

- Q0 paired forward/replay 预计 `0.1–0.3 GPU-hour`；Q1/final Router 数分钟。
- 无教师重训、baseline 重跑、新数据或模型多种子。
- TDD RED→GREEN → remote preflight → Q0 → 条件式 Q1 → 条件式 dev。
