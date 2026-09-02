# Round 1 Refinement

## Problem Anchor

- Bottom-line problem: 在 RGBNT201 的固定 `141-fit / 30-dev` 协议上，把 CNN、Transformer、Mamba 三类异构专家已经被 V12 证明存在的互补空间转化为可部署融合增益，同时严格保留 exact Signal baseline，最终 seed 42 fused mAP 达到至少 65，并严格超过 baseline 与全部固定专家。
- Must-solve bottleneck: V12 的身份 Router 在完整路径 identity-OOF 条件下仍不能跨折泛化；learned expected margin 为 `-0.117330`，低于固定策略 `-0.099975`，Top-slot accuracy 为 `12.2592%`，低于 majority `16.8126%`，而质量响应门已经通过。
- Non-goals: 不复现 baseline；不做多种子；主结果达到 65 mAP 前不做消融；不访问 official test；不以增加另一套大 backbone、HFER 或超参数扫描掩盖 Router 监督失配。
- Constraints: 所有训练、推理和数据处理仅在远端单张 RTX 3090 24 GiB 上进行；固定 seed 42、B64/K8、既有 3-fold identity-OOF 划分和 V8/V12 已登记超参数；复用现有 V12 fold checkpoints，不重新训练 Signal baseline 或 V12 教师。
- Success condition: train-only Q0 证明实际融合路径上的九槽反事实效用非退化且具有专家/模态互补；Q1 learned Router 的 held-out-fold expected counterfactual utility 严格超过固定策略、Top-slot accuracy 严格超过 majority，并继续通过三模态退化响应与 missing-mass=0；只有全部通过才允许一次 final-only dev，且 fused≥65 mAP 并严格超过 exact Signal、V8 Phase-B 和三个固定专家。

## Anchor Check

- Original bottleneck: V12 已有专家互补，但 Router 的 residual-only target 与部署融合路径不一致，且 Q1 输入并不等于 final all-fit inference 输入。
- Why the revised method still addresses it: OOF 模型只作为不泄漏的 utility teacher；Router 从训练、验证到 final inference 始终读取同一个冻结 all-fit Phase-A 特征坐标，输出再回放到 OOF 路径验证真实检索收益。
- Reviewer suggestions rejected as drift: 不加入 DINO、HFER、新 backbone、多种子或超参数扫描；这些不修复 target/input/deployment 对齐。

## Simplicity Check

- Dominant contribution after revision: Deployment-Aligned Fusion-Path Counterfactual Distillation。
- Components removed or merged: 删除 Gram relational descriptor；删除 learned alpha，V13 固定 `alpha=0.2`；quality 只保留为安全门。
- Reviewer suggestions rejected as unnecessary complexity: 不另建 all-fit expert、适配器或 domain-alignment network。
- Why the remaining mechanism is still the smallest adequate route: 新增的只有离线 target/cache 生成和现有 Router 的输入配对；backbone、专家、Router heads 与部署模型均复用。

## Changes Made

### 1. Teacher、student 与 deployment 坐标闭合

- Reviewer said: Q1 fold teacher 输入与 final all-fit 输入分布不一致，是最大 drift 风险。
- Action: OOF checkpoint 只产生 target 与 non-saturated evaluation path；Router 始终读取同一冻结 V8 Phase-A all-fit 模型对相同样本产生的 direct/residual 特征。
- Reasoning: 这样 three-fold Router heldout identity validation 与 final refit/dev 使用同一个 feature generator；不再要求 MLP 跨三个 fold raw basis 泛化。
- Impact on core method: 形成 paired cache：`teacher_oof_path + deployment_student_input`。

### 2. 删除 Gram descriptor 并固定 alpha

- Reviewer said: Gram 只能保证共同正交变换不变，不能声称解决任意跨折漂移；learned alpha 还会引入 teacher/path 不一致。
- Action: 删除 Gram 组件；固定 train/Q0/Q1/final 的 `alpha=0.2`。
- Reasoning: 同一个 all-fit input generator 已直接消除坐标来源变化；固定 alpha 使 counterfactual target 与部署融合完全同参。
- Impact on core method: 一个主贡献、一个小 Router、零新 backbone、零新增 loss。

### 3. 完整定义 reference bank、margin 与 removal

- Reviewer said: bank、符号、归一化和 hash 不够具体。
- Action: 在下述 Revised Proposal 中给出可执行定义和 per-fold fail-fast gate。
- Reasoning: 只有 teacher target 与 evaluator 共享同一数学函数，才能称路径一致。
- Impact on core method: Q0/Q1 可以用 worked-example TDD 和真实 cache receipt 独立验证。

## Revised Proposal

# Research Proposal: TriFusion V13 部署对齐的融合路径反事实蒸馏

## Problem Anchor

- Bottom-line problem: 在 RGBNT201 的固定 `141-fit / 30-dev` 协议上，把 CNN、Transformer、Mamba 三类异构专家已经被 V12 证明存在的互补空间转化为可部署融合增益，同时严格保留 exact Signal baseline，最终 seed 42 fused mAP 达到至少 65，并严格超过 baseline 与全部固定专家。
- Must-solve bottleneck: V12 的身份 Router 在完整路径 identity-OOF 条件下仍不能跨折泛化；learned expected margin 为 `-0.117330`，低于固定策略 `-0.099975`，Top-slot accuracy 为 `12.2592%`，低于 majority `16.8126%`，而质量响应门已经通过。
- Non-goals: 不复现 baseline；不做多种子；主结果达到 65 mAP 前不做消融；不访问 official test；不以增加另一套大 backbone、HFER 或超参数扫描掩盖 Router 监督失配。
- Constraints: 所有训练、推理和数据处理仅在远端单张 RTX 3090 24 GiB 上进行；固定 seed 42、B64/K8、既有 3-fold identity-OOF 划分和 V8/V12 已登记超参数；复用现有 V12 fold checkpoints，不重新训练 Signal baseline 或 V12 教师。
- Success condition: train-only Q0 证明实际融合路径上的九槽反事实效用非退化且具有专家/模态互补；Q1 learned Router 的 held-out-fold expected counterfactual utility 严格超过固定策略、Top-slot accuracy 严格超过 majority，并继续通过三模态退化响应与 missing-mass=0；只有全部通过才允许一次 final-only dev，且 fused≥65 mAP 并严格超过 exact Signal、V8 Phase-B 和三个固定专家。

## Technical Gap

V12 的 target 是 `slot residual` 单独检索的绝对 margin，部署输出却是 exact Signal 与归一化 routed residual bank 的拼接；它没有测量一个 slot 在真实融合中的 signed marginal effect。V12 Q1 又直接读取各 OOF fold 模型的 raw feature，而 final Router 会读取 all-fit V8 Phase-A feature，导致训练输入坐标与部署输入坐标不一致。

最小修复不是继续加工 fold feature，而是严格分离 teacher 与 student：OOF model 只负责产生非泄漏 target 和非饱和检索评价；Router 从第一步起就读取最终部署的 all-fit Phase-A feature。expert/modality action 的九个语义槽在两边按固定名称对应。

## Method Thesis

- One-sentence thesis: 将 complete-path identity-OOF teacher 的真实融合贡献效用蒸馏到始终读取 deployment all-fit feature 的层级 Router，并把其决策回放到 OOF 路径验证，从而同时对齐 target、输入坐标和最终推理接口。
- Why this is the smallest adequate intervention: 复用所有 checkpoint、专家和 Router，只生成 paired cache，固定 alpha，不增加任何 backbone 或新 loss。
- Why this route is timely in the foundation-model era: 强 CLIP/Signal 模型作为冻结部署表示，identity-OOF complete model 作为离线 counterfactual teacher，小 Router 学控制策略。

## Contribution Focus

- Dominant contribution: Deployment-Aligned Fusion-Path Counterfactual Distillation。
- Optional supporting contribution: 无；paired cache、回放评价和 quality gate 都是主贡献的必要合同。
- Explicit non-contributions: hierarchical routing、quality degradation、exact prefix 和 bounded/fixed residual energy 是沿用机制；本阶段不声称 HFER、校准因果性、official 或 SOTA。

## Proposed Method

### Complexity Budget

- Frozen / reused backbone: V8 Phase-A all-fit deployment model；V12 三个 complete-path OOF teacher；exact Signal baseline。
- New trainable components: 仅现有层级 Router。
- Tempting additions intentionally not used: Gram descriptor、learned alpha、DINO、HFER、额外 backbone、joint fine-tuning、超参数扫描。

### Exact Fusion Function

令有效槽集合为 `S(x)`，九槽 residual 为 `r_s(x)∈R^512`，exact Signal 为 `z0(x)∈R^3072`。当前代码的 bank 是 blockwise concat，不是向量求和：

`b(x,w) = vec({w_s r_s(x)} for s in fixed expert-major/modality-minor order)`

`F(x;w) = L2(concat(z0(x), 0.2 * ||z0(x)||_2 * L2(b(x,w))))`。

最终 `L2` 与 frozen evaluator 一致；输出进入最终 `L2` 前的前 3072D 必须逐元素等于 `z0`。V13 不学习 alpha。

### Fixed OOF Reference Bank

对 fold `f`：

1. `Q_f` 是该 fold 中所有具备不同 camera 正样本的 held-out fit records，数量沿用 V12 的 `190/179/202`；`G_f` 是相同有序 records 的复制，因此 query 自身和同 identity/same camera 样本按既有 evaluator 规则剔除。
2. 所有 clean 样本三模态有效，`w^U_s=1/9`。
3. `B_f={F_f(x_i;w^U):x_i∈G_f}`，只用 fold OOF checkpoint 生成一次。
4. 对 query `q`，positive 是 `same identity && different camera`；negative 是 `different identity`。
5. 距离是两个最终 L2 embedding 的 Euclidean distance；higher-is-better margin 明确定义为：

   `M_f(q,B_f)=min_negative_distance - max_positive_distance`。

6. bank receipt 绑定：有序样本路径 SHA、identity/camera tensors SHA、fold Signal/expert checkpoint SHA、baseline/residual tensors SHA、fusion/evaluator/config/commit SHA、`alpha=0.2`、slot order 和 mask。

### Query-Side Counterfactual Utility

完整 query margin 为 `M_f(F_f(q;w^U),B_f)`。对有效槽 `s`，先在未归一化 block bank 中把 `s` 权重置零，再把其余八槽设为 `1/8`，完整重跑 residual-bank L2、concat 和 final L2：

`F_f^{-s}(q)=F_f(q;w^{-s})`。

reference bank `B_f` 保持逐元素不变。signed target 为：

`Delta_f(q,s)=M_f(F_f(q;w^U),B_f)-M_f(F_f^{-s}(q),B_f)`。

因此 `Delta>0` 严格表示保留该槽提高 margin，`Delta<0` 表示该槽有害。

### Paired Teacher–Student Cache

对同一个 ordered `Q_f/G_f` 额外用冻结 all-fit V8 Phase-A deployment model 前向一次，存储：

- `student_direct_modal`；
- `student_modal_residual`；
- 同一行的 `teacher_oof_baseline/modal_residual/Delta`；
- identity、camera、fold、sample hash。

Router train、heldout-fold inference 和 final refit 全部只读取 `student_*`，从而与 final dev 的输入生成器逐 checkpoint SHA 一致。训练 target 始终是对应行的 `teacher Delta`，避免用 all-fit identity leakage 生成监督。

### Router and Evaluation Replay

Router 保留 `P(m|x)P(e|m,x)` 和 missing-mask 语义；alpha 输出改为常量 `0.2`。Q1 每次 heldout-fold 验证时：

1. Router 从 `student_*` 产生 query/gallery weights；
2. 这些 weights 按同一 sample key 回放到 `teacher_oof_baseline/modal_residual`；
3. 使用 exact `F_f` 和同一 ReID filtering 计算 non-saturated mean margin、mAP、Rank-1；
4. fixed policy 只由 training folds 选择，不能用 heldout fold 选择 slot。

这使 Router 输入与 deployment 一致，而检索评价仍位于真正未见 identity 的 OOF teacher 路径。

### Loss and Training

- `target_weights=softmax(masked Delta/0.05)`；Router 权重用现有 KL。
- 删除 alpha head loss；V13 全程 `alpha=0.2`。
- 受控退化的 modality-quality KL 保持权重 `1.0`，只作为 safety gate。
- 三折各 100 epoch、LR `3.5e-4`、seed42；不改 optimizer 或 hidden width。

### Q0 Health and Transfer Gates

每 fold 必须报告：九槽 signed Delta mean/std/min/max、正值比例、unique positive winners、target entropy、full/remove embedding 非恒等检查、bank immutability SHA。每个 expert 和 modality 在聚合后必须至少有一个 unique positive winner，slot Oracle mean Delta 必须严格超过 best fixed slot。

另报告 OOF Delta 与 all-fit 同构路径 Delta 的 target-transfer audit，但 all-fit Delta 不进入训练：OOF oracle 选出的 slot 在 all-fit Delta 上必须 aggregate 严格优于由 train folds 选出的 fixed slot，且各 fold 不劣于 fixed（容差 `1e-6`）。否则说明 action semantics 不转移，Q1 禁止。

### Q1 Fail-Fast Gate

- 每 fold learned expected OOF Delta 不低于 train-fold fixed policy（容差 `1e-6`），aggregate gain 必须 `>1e-4`。
- 每 fold Top-slot accuracy 不低于 majority，aggregate 必须严格高于 majority。
- 将 student weights 回放到 OOF path 后，每 fold mean identity margin 不低于 fixed replay（容差 `1e-6`），aggregate gain 必须 `>1e-4`。
- RGB/NI/TI corruption 后本模态 mass 均下降；missing mass 严格为 0。
- 任一失败：不 final refit、不 dev、不扫参数。

### Final Integration

Q1 全门通过后，在全部 571 paired rows 上 final refit Router；组合的 checkpoint 必须绑定与 `student_*` 完全相同的 all-fit Phase-A checkpoint SHA。只进行一次 30-dev frozen evaluation，alpha 保持 0.2，exact Signal prefix 不变。

### Failure Modes and Diagnostics

- Q0 action transfer fail：OOF teacher 槽效用不能转移到 deployment expert semantics；封存 V13。
- Q1 expected Delta 或 replay margin fail：deployment feature 无法预测 OOF utility；封存 V13。
- Q1 过门但 dev<65：只支持 train-only routing 结论，不授权 official/消融/SOTA。

### Novelty and Elegance Argument

V13 把原本混在一起的“生成 target”和“提供 Router 输入”拆开：身份隔离的 complete model 只提供可信 utility，最终 all-fit model只提供部署坐标，sample-key pairing 把两者连接；所有决策再回放到 OOF 路径检验真实检索结果。它以一张 paired cache 修复 target/path/coordinate 三重失配，而不是增加网络容量。

## Claim-Driven Validation Sketch

### Claim 1: deployment-aligned input 可以学习 actual-path OOF utility

- Minimal experiment: Q0 paired cache + 单次固定 Q1。
- Baselines / ablations: 仅预注册 fixed policy/majority；主结果前不做消融。
- Metric: per-fold/aggregate expected Delta、Top1、OOF replay margin、quality/missing gates。
- Expected evidence: 所有 per-fold non-inferiority 与 aggregate strict-gain 门通过。

### Claim 2: 该 Router 把互补转为可部署主结果

- Minimal experiment: Q1 通过后的一次 seed42 dev。
- Baselines / ablations: exact Signal `58.0109`、V8 Phase-B `58.4050`、三个固定专家。
- Metric: mAP / Rank-1。
- Expected evidence: fused≥65 mAP 且严格超过全部登记输出。

## Experiment Handoff Inputs

- Must-prove claims: target/path 同义；student/deployment 同坐标；heldout identity OOF replay 真增益。
- Must-run ablations: 主结果达到 65 后，只做 `actual-path→residual-only` 与 `paired deployment input→fold raw input` 两项 claim-critical 删除检查。
- Critical datasets / metrics: RGBNT201 141-fit OOF；30-dev 条件式一次；official 禁止。
- Highest-risk assumptions: OOF teacher 的 expert/modality action semantics 能转移到 all-fit deployment experts。

## Compute & Timeline Estimate

- Estimated GPU-hours: 现有 V12 checkpoints 和 all-fit Phase-A 各前向一次，预计 0.1–0.3 GPU-hour；Router Q1 与条件式 final/dev 数分钟。
- Data / annotation cost: 0。
- Timeline: TDD RED→GREEN、远端 preflight、Q0；仅 Q0 通过后 Q1；仅 Q1 通过后 dev。
