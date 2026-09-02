# Research Proposal: TriFusion V13 路径一致的关系型反事实路由

## Problem Anchor

- Bottom-line problem: 在 RGBNT201 的固定 `141-fit / 30-dev` 协议上，把 CNN、Transformer、Mamba 三类异构专家已经被 V12 证明存在的互补空间转化为可部署融合增益，同时严格保留 exact Signal baseline，最终 seed 42 fused mAP 达到至少 65，并严格超过 baseline 与全部固定专家。
- Must-solve bottleneck: V12 的身份 Router 在完整路径 identity-OOF 条件下仍不能跨折泛化；learned expected margin 为 `-0.117330`，低于固定策略 `-0.099975`，Top-slot accuracy 为 `12.2592%`，低于 majority `16.8126%`，而质量响应门已经通过。
- Non-goals: 不复现 baseline；不做多种子；主结果达到 65 mAP 前不做消融；不访问 official test；不以增加另一套大 backbone、HFER 或超参数扫描掩盖 Router 监督失配。
- Constraints: 所有训练、推理和数据处理仅在远端单张 RTX 3090 24 GiB 上进行；固定 seed 42、B64/K8、既有 3-fold identity-OOF 划分和 V8/V12 已登记超参数；复用现有 V12 fold checkpoints，不重新训练 Signal baseline 或 V12 教师。
- Success condition: train-only Q0 证明实际融合路径上的九槽反事实效用非退化且具有专家/模态互补；Q1 learned Router 的 held-out-fold expected counterfactual utility 严格超过固定策略、Top-slot accuracy 严格超过 majority，并继续通过三模态退化响应与 missing-mass=0；只有全部通过才允许一次 final-only dev，且 fused≥65 mAP 并严格超过 exact Signal、V8 Phase-B 和三个固定专家。

## Technical Gap

V12 已排除“三专家没有互补性”：完整特征路径 identity-OOF 下，CNN/Transformer/Mamba residual 的 hard Oracle 比最强固定专家高 `5.3130 mAP`，三专家与三模态都有独有胜例。失败集中在 Router。

当前监督却不是部署效用。`build_v12_complete_path_oof_targets.py` 对每个 slot 调用 `_margins_from_features(slot_residual)`，得到 residual-only 的绝对 identity margin；部署输出则由 `OOFMarginRoutedFusion` 构成 `[z0, alpha·||z0||·Normalize(weighted residual bank)]`。因此当前 target 没有测量某一贡献在 exact baseline 和真实 residual-bank 归一化下究竟增加还是破坏检索 margin。V6 终态也已经记录过相同的“residual-only target 与最终拼接表示边际收益不一致”。

第二个已观测风险是三个 V12 fold 的 slot 均值和 winner 分布明显漂移，同时现有 Router 直接对各 fold 独立训练所得的 512D 坐标做线性投影。仅看范数无法区分这种折间语义坐标变化。直接加深 MLP、调温度或延长训练不会修正 target 的含义，也不会让原始坐标天然可比较。

候选路线比较：

- Route A（最小路线）：只把 residual-only margin 换成部署路径 counterfactual utility，保留原始坐标 Router。优点是修改最少；缺点是仍要求一个 MLP 跨三个独立 fold teacher 的 512D 坐标泛化。
- Route B（选定路线）：同样使用部署路径 counterfactual utility，但把 Router 输入改成由 direct/residual 的归一化 Gram 关系和 direct norm 构成的固定语义描述。它只增加一个无参数描述变换，不增加 backbone，直接同时对齐 target 语义与跨折输入语义。

Route B 是当前最小充分修复。若其 train-only 门失败，证据将否定这条路，而不是触发 epoch、LR、温度或 alpha 扫描。

## Method Thesis

- One-sentence thesis: 用 actual deployed fusion path 的 identity-OOF 反事实边际效用监督一个坐标关系型层级 Router，使异构专家互补从 residual-only Oracle 诊断转化为可泛化、质量感知且有界的融合决策。
- Why this is the smallest adequate intervention: 保留 V8 专家、层级 `P(m|x)P(e|m,x)`、质量损失、alpha 上界和 exact Signal 前缀，只替换错误的效用定义与不可跨折比较的输入表示。
- Why this route is timely in the foundation-model era: 它把强 CLIP/Signal backbone 当作冻结专家场，用 OOF teacher 的反事实效用蒸馏控制策略，而不是再训练一个平行大模型。

## Contribution Focus

- Dominant contribution: Fusion-Path Counterfactual Utility Distillation——在固定完整 reference bank 上，以真实 `[baseline, routed residual bank]` 路径的贡献移除效应训练 Router，而非 residual-only 代理。
- Optional supporting contribution: Fold-Relational Router Descriptor——使用 12 个语义向量的 Gram 关系和 direct norm 构成无参数、全局正交变换不变的输入，使三个 OOF teacher 与 final all-fit 模型共享可解释坐标。
- Explicit non-contributions: 不把既有层级归一化、quality degradation、bounded alpha 或 exact baseline prefix 重新包装成新贡献；不在本阶段声称 HFER、校准因果性、official SOTA 或跨数据集泛化。

## Proposed Method

### Complexity Budget

- Frozen / reused backbone: exact Signal 3072D baseline；V8 CNN/Transformer/Mamba pretrained-tail residual experts；V12 三个 complete-path OOF Signal/expert checkpoints；现有质量退化生成器。
- New trainable components: 仍只有一个小型层级 Router；不增加第二个 trainable network。
- Tempting additions intentionally not used: DINO、HFER、额外 backbone、MoE top-k、temperature/LR/epoch/alpha scan、joint expert fine-tuning。

### System Overview

```text
V12 fold checkpoint + held-out fit identity
        ↓
exact Signal z0 + 3×3 residual slots
        ↓
固定 uniform full-fusion query / 固定 full reference bank
        ↓ 逐槽 do(c_e,m = 0)
Δ_e,m = margin(full query) - margin(slot-removed query)
        ↓
不可训练关系描述 Φ(direct, residual)
        ↓
层级 Router: P(m|x) · P(e|m,x), bounded alpha
        ↓
[exact z0, alpha·||z0||·Normalize(weighted residual bank)]
```

### Core Mechanism

- Input / output: 输入仍为 `direct_modal: B×3×512`、`modal_residual: B×3×3×512` 和 bool mask；输出保持 `weights/modal_probabilities/expert_probabilities/alpha` 公共合同。
- Counterfactual target: 对每个 fold 的 held-out query，先以固定 uniform 九槽权重和预注册 `alpha=0.2` 生成完整 query；reference/gallery bank 只计算一次并在九次 query-side intervention 中保持不变。对槽 `s=(e,m)`，在完整融合路径把该 query 槽置零并对剩余有效槽重新归一，定义

  `Δ_q,s = identity_margin(full_fused_query, fixed_reference_bank) - identity_margin(remove_s_query, fixed_reference_bank)`。

  `Δ>0` 表示该槽对真实融合有帮助，`Δ<0` 表示有害。目标权重继续使用固定 temperature 的 masked softmax；alpha target 仅由最佳正 `Δ` 生成，所有槽均无益时允许趋近 0。
- Relational descriptor: 将 3 个 direct 向量与 9 个 residual 向量按固定顺序组成 `V∈R^(12×512)`，对有效行 L2 归一后计算 `G=VV^T`。每个模态或 slot 使用其对应 Gram row，加上三个 direct log-norm 和固定 expert/modality ID embedding。该描述在所有 512D 向量共同经历正交坐标变化时保持不变；mask 对应行列严格清零。
- Architecture or policy: 复用当前共享 modal head、expert head、alpha head 的层级输出语义，只把 raw projection 输入换成共享 relational row encoder。
- Training signal / loss: 保持现有 counterfactual target KL、alpha MSE 和 controlled-degradation modality-quality KL，不改权重、epoch、LR 或温度。
- Why this is the main novelty: teacher 与 student 首次在“同一真实融合路径、可跨 OOF teacher 比较的关系空间”上对齐；它不是再增加专家，而是修复 Oracle 无法被 Router 学到的机制缺口。

### Optional Supporting Component

- Only include if truly necessary: 关系描述与主机制一起进入 Q1，因为现有三折 target 均值和 winner 分布明显不同，且 final 部署模型不是任一 fold teacher。
- Input / output: 不改变 Router 公共 seam，只改变内部特征化。
- Training signal / loss: 不增加新 loss。
- Why it does not create contribution sprawl: 它是反事实监督能够跨折迁移的表示接口，不是并列任务或新网络。

### Modern Primitive Usage

- Which foundation-model-era primitive is used: frozen CLIP/Signal representation 与 OOF teacher-student utility distillation。
- Exact role in the pipeline: CLIP/Signal 提供强共享语义和 exact safety floor；OOF teacher 只离线产生部署路径反事实效用，Router 是轻量 student。
- Why it is more natural than an old-school alternative: 在单卡 3090 上复用强预训练表示并蒸馏控制策略，比训练三套完整 backbone 或扩大 Router 容量更直接。

### Integration into Base Generator / Downstream Pipeline

V13 Q0 直接读取现有 V12 fold Signal/expert checkpoint，不重新训练教师。只重新前向 held-out fit records，生成含 query-side full-intervention delta、固定 bank hash、fold/checkpoint/code hash 的新 cache。Q1 冻结全部专家，用相同 3-fold Router 验证；通过后才在全部 OOF rows 上 final refit，并与既有 all-fit V8 Phase-A 专家组合。所有 fused embedding 的前 3072D 必须逐元素等于 exact Signal baseline。

### Training Plan

1. Train-only Q0：生成 actual-path counterfactual cache；检查 571 queries、3 folds、身份零交集、每槽 signed delta、专家/模态独有正效用、fixed-bank hash、dev0/official0。
2. Train-only Q1：三个 Router fold 各固定 100 epoch；保持 seed42、LR `3.5e-4`、temperature `0.05`、alpha max/init `0.5/0.2`、quality weight `1.0`。
3. Gate：learned expected `Δ` 严格高于 train-fold fixed policy，Top-slot accuracy 严格高于 majority，三模态 corruption mass 均下降，missing mass 严格为 0。
4. 仅 Q1 通过后：all-row final refit，单次 frozen dev；不训练专家、不打开 HFER。

### Failure Modes and Diagnostics

- Counterfactual teacher 仍非退化但 Router 不超过 fixed policy：说明 query-local relation descriptor不足以预测 gallery-relative效用；立即封存 V13，不扫描超参数。
- Exact-path delta 没有三专家/三模态互补：说明 V12 residual-only Oracle 不能转化到真实融合路径；Q0 停止，不训练 Router。
- Q1 通过但 dev<65 或不严格胜出：方法只支持 train-only routing claim，不授权 official、消融或 SOTA。

### Novelty and Elegance Argument

现有仓库方法已经提出 OOF full intervention 的思想，但 V8/V12 实现实际退化成 residual-only absolute margin。V13 的贡献不是命名新模块，而是把离线 teacher、Router 输入、最终检索路径三者在一个可执行合同上闭合：固定 reference bank、真实 residual-bank 归一化、逐槽 query-side removal、signed marginal utility、折间关系表示。新增计算只发生在离线 Q0；推理仍是一个小 Router。

## Claim-Driven Validation Sketch

### Claim 1: 路径一致反事实效用是可学习的 train-only routing target

- Minimal experiment: 复用 V12 三折 checkpoints 生成 571-query actual-path cache，并运行一个固定配置的三折 Router Q1。
- Baselines / ablations: 只比较预注册 fixed-slot/majority policy；主结果过门前不做 residual-only target 消融。
- Metric: held-out-fold expected `Δ`、Top-slot accuracy、quality mass response、missing mass。
- Expected evidence: learned strictly > fixed、Top1 strictly > majority、quality gates 全通过。

### Claim 2: 该控制策略能把互补转成可部署主结果

- Minimal experiment: Q1 通过后仅一次 seed42 fixed dev。
- Baselines / ablations: exact Signal `58.0109`、当前 deployable best V8 Phase-B `58.4050`、三个固定专家。
- Metric: mAP / Rank-1，主门以 mAP 为准。
- Expected evidence: fused≥65 mAP 且严格超过所有登记比较对象。

## Experiment Handoff Inputs

- Must-prove claims: actual-path target 非退化且可跨身份折学习；final Router 保持质量语义并产生部署增益。
- Must-run ablations: 主结果达 65 后才安排 old residual-only target 与 raw-coordinate Router 两项删除检查。
- Critical datasets / metrics: RGBNT201 固定 141-fit train-only OOF；30-dev 只在 Q1 过门后一次；official test 禁止。
- Highest-risk assumptions: query-side relation descriptor是否包含足够信息预测固定 reference bank 上的身份效用。

## Compute & Timeline Estimate

- Estimated GPU-hours: Q0 只复用 checkpoint 前向，预计 0.1–0.25 GPU-hour；Q1 约 1 分钟；通过后的 final refit+dev 预计数分钟。实现与测试预计 1–2 小时，不占 GPU。
- Data / annotation cost: 0；只用现有 RGBNT201 fit identity 标签。
- Timeline: 先完成公共 seam 的 RED→GREEN 与远端相邻测试，再运行 Q0；Q0/Q1 任一失败即终止。
