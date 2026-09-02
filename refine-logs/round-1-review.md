# Round 1 Review

<details>
<summary>GPT-5.5 xhigh 原始评审</summary>

## 原始评审

### 总体判断

V13 的方向比前几版更像一个可审稿的 method-first proposal：它没有继续堆 backbone，也没有把失败归因给训练轮数，而是抓住了 V12 的真实瓶颈：Router 监督目标和最终部署路径不一致。主贡献“Fusion-Path Counterfactual Utility Distillation”是清楚的，也保持了 exact Signal baseline 这个 Problem Anchor。

但当前版本还不能算方法闭合。最关键的问题是：Q0 teacher、Q1 Router 验证、final all-fit inference 三者仍可能不是同一个分布/同一个函数路径。proposal 说的是“路径一致”，但几个定义还不够数学化，尤其是 fixed reference bank、margin 符号方向、slot removal 后归一化、以及 fold-relational descriptor 是否真的解决跨 fold 坐标漂移。

### Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 8 |
| Method Specificity | 6 |
| Contribution Quality | 7 |
| Frontier Leverage | 7 |
| Feasibility | 7 |
| Validation Focus | 6 |
| Venue Readiness | 5 |

**OVERALL SCORE = 6.75 / 10**  
权重计算：`8*.15 + 6*.25 + 7*.25 + 7*.15 + 7*.10 + 6*.05 + 5*.05 = 6.75`

### 低分项与修复

**Method Specificity: 6 / 10**

Priority: CRITICAL

弱点：

1. `fixed_reference_bank` 没有被完全定义。现在只说“只计算一次并保持不变”，但没有明确 bank 的样本集合、特征路径、alpha、uniform 权重、mask、归一化、正负样本定义、是否排除 query 自身、是否遵守 cross-camera 规则，以及 hash 绑定什么。
2. `identity_margin` 的方向不够安全。`Δ = margin(full) - margin(remove)` 只有在 margin 是 higher-is-better 时才成立。如果底层沿用 distance margin 或 loss-style margin，方向会反。
3. slot removal 后归一化写了“重新归一”，但需要明确是“置零 slot 后重新走完整 fusion 函数”，不是从已经 normalized 的 residual bank 里减分量。
4. relational Gram descriptor 只能保证“同一样本内 12 个向量共同经历同一个正交变换”时不变。它不能保证解决 fold-specific 非正交漂移、尺度漂移、专家语义重排或 final all-fit 与 OOF fold checkpoint 的分布差异。
5. Q1 使用 V12 fold checkpoints 的 held-out fold 分布；final refit/dev 又说与 all-fit V8 Phase-A 专家组合。这个地方存在输入分布不一致风险。

方法层面的具体修复：

- 在 proposal 中显式定义：

  `F(x; w, alpha) = L2([z0(x), alpha * ||z0(x)|| * L2(sum_s w_s r_s(x))])`

  `B_f = {F(x_i; uniform-valid, 0.2)}`，并固定样本索引、label mask、positive/negative rule、feature tensor hash、checkpoint hash。
- 定义 higher-is-better margin，例如 `M(q,B)=mean_pos_sim(q)-max_neg_sim(q)` 或明确已有函数等价于 higher-is-better；如果用 distance，必须改成 `M = min_neg_distance - mean_pos_distance`。
- removal 必须写成：

  `F_remove_s(q)=F(q; uniform-valid-over-S\{s}, 0.2)`

  reference bank 不变，slot removal 只发生在 query side。
- 不要声称 Gram descriptor “解决”跨 fold 坐标，只能声称它“减少 raw-coordinate dependence”；真正解决与否由 Q1 fold-generalization gate 判定。
- final all-fit 前增加一个 train-only compatibility check：确认 final all-fit descriptor/slot-delta 分布与三折 OOF descriptor/slot-delta 分布没有明显失配，或者改成 final refit 使用与 Q1 完全同构的 all-fit complete-path expert state。

**Validation Focus: 6 / 10**

Priority: IMPORTANT

弱点：

1. “strictly exceeds fixed / majority”太弱，可能被数值噪声或单 fold 偶然性通过。
2. 主结果前不做 ablation 是合理约束，但 Q0/Q1 必须提供足够 fail-fast 证据；当前 gate 主要是 aggregate，没有明确 per-fold 要求。
3. 若 Q1 通过但 dev 失败，当前设计仍难以判断失败来自 target、descriptor、alpha、还是 all-fit distribution shift。

方法层面的具体修复：

- Q1 gate 不应只看 aggregate。要求每个 fold learned expected Δ 都不低于 fixed，整体严格高于 fixed，并且差值超过数值容差。
- Top-slot accuracy 也要求 per-fold 不劣于 majority，aggregate 严格高于 majority。
- Q0 需要输出 target health：每槽 signed Δ 分布、正 Δ 比例、每专家/模态 unique positive utility、target entropy、uniform full vs remove_s 的 embedding/margin sanity check。
- 这些不是 ablation，不违反“主结果前不做消融”；它们是 teacher/cache/gate 的健康检查。

**Venue Readiness: 5 / 10**

Priority: IMPORTANT

弱点：

1. 目前仍是早期 proposal，没有主结果。
2. 即使 dev≥65，论文级 claim 仍需要两个最小删除检查：old residual-only target 与 raw-coordinate Router。proposal 已把它们放在主结果后，这个顺序可以接受。
3. “single seed only”可以作为内部开发约束，但顶会投稿层面会被质疑稳定性。若最终只报 seed42，venue readiness 会受限。

方法层面的具体修复：

- 当前阶段不扩大实验菜单；先把 Q0/Q1 定义闭合。
- 若主结果成功，再只做两项 claim-critical ablation：`actual-path target -> residual-only target`，`relational descriptor -> raw coordinate`。
- 不要提前引入 DINO/HFER/extra backbone/multi-seed 来掩盖方法不闭合。

### 指定问题检查

- **固定 reference bank 定义**：目前不足。必须具体绑定 bank 样本、特征生成函数、alpha、uniform mask、label/margin rule 和 hash。
- **query-side removal 符号/方向**：方向在 higher-is-better margin 下正确；如果底层 margin 是 distance/loss，需要反向。
- **slot removal 后归一化**：proposal 意图正确，但需要写成“remove before weighted-sum and residual normalization, then rerun fusion path”。
- **关系描述是否解决跨 fold 坐标**：只能部分解决。Gram 不变性不足以覆盖 fold checkpoint 的任意语义漂移。
- **Q1 与 final all-fit 输入分布一致性**：当前最危险。V12 fold teacher 到 all-fit V8 Phase-A final path 之间必须证明同构，否则 Q1 过门不强支持 final dev。
- **不在主结果前做消融是否够 fail-fast**：可以，但 Q0/Q1 gate 要更硬，尤其 per-fold 和非噪声级 improvement。

### Simplification Opportunities

1. 把 “Fold-Relational Router Descriptor” 从 optional supporting contribution 改成主贡献的一部分；不要包装成第二贡献。
2. 为了保持路径一致，优先固定 `alpha=0.2` 完成第一版 V13 gate；若保留 alpha head，则必须说明 teacher Δ 和 learned alpha 的一致性。
3. quality degradation 只作为沿用的安全检查，不要放进 novelty 叙事。

### Modernization Opportunities

NONE for additional modules.  
当前 foundation-model-era 使用方式已经合适：冻结 CLIP/Signal 表征场，做轻量 utility distillation。不要加 DINO、LLM、HFER 或新 backbone。

### Drift Warning

存在轻度 drift 风险：proposal 的目标是“把 V12 OOF 互补转成 deployable fusion gain”，但 final all-fit 阶段如果改用与 Q1 teacher 不同分布的 V8 Phase-A all-fit experts，就从已验证的 OOF complete-path routing problem 漂移到了另一个输入分布。需要在 proposal 中闭合这个路径。

### Verdict

**REVISE**

核心方向值得继续，不需要 RETHINK；但必须先把 teacher target、reference bank、margin convention、removal normalization、relational descriptor claim 和 final all-fit 分布一致性写到可执行级别。Ready 条件目前不满足。

</details>
