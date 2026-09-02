# Round 2 Review

<details>
<summary>GPT-5.5 xhigh 原始评审</summary>

## Round 2 原始评审

### 总体判断

Revised Proposal 明显比 Round 0 更强。最重要的修正是删掉 Gram relational descriptor，把问题从“跨 fold 坐标不变性”改成更干净的 teacher/student 分离：OOF teacher 只产非泄漏 utility target 和 replay evaluator；Router 从训练、OOF 验证到 final dev 始终读取同一个冻结 all-fit V8 Phase-A deployment feature。这个修改更简单，也更符合 Problem Anchor。

当前主贡献更聚焦：不是“反事实 target + 关系描述”两个贡献并行，而是一个 Deployment-Aligned Fusion-Path Counterfactual Distillation 合同。fixed bank、margin 方向、query-side removal-before-normalization、blockwise concat、alpha fixed=0.2、provenance hash 都已经补到可实现级别。

主要剩余风险不是方法复杂度，而是科学边界和 gate 强度：all-fit student feature 已见过全部 141 fit identities，因此 Q1 不能被表述成“完整 identity-OOF representation generalization”。它只能支持更窄但仍有用的 claim：在最终部署特征接口上，Router 是否能从 train-fold identities 学到 OOF teacher utility，并把 student decision 回放到 identity-heldout OOF retrieval path 后仍产生非饱和收益。

### Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9 |
| Method Specificity | 8 |
| Contribution Quality | 8 |
| Frontier Leverage | 8 |
| Feasibility | 8 |
| Validation Focus | 7 |
| Venue Readiness | 6 |

**OVERALL SCORE = 8.00 / 10**

权重计算：`9*.15 + 8*.25 + 8*.25 + 8*.15 + 8*.10 + 7*.05 + 6*.05 = 8.00`

### Dimension Notes

**Problem Fidelity: 9**

Problem Anchor 基本 preserved。它仍然围绕 V12 互补空间、Router 监督失配、exact Signal baseline、single seed dev gate、no official/no pre-main ablation。没有明显 problem drift。

唯一边界：Q1 不再是纯粹的 complete-path identity-OOF feature validation，因为 Router 输入来自 all-fit Phase-A feature。只要论文/文档把它称为 “deployment-feature policy validation with OOF teacher/replay labels”，而不是“heldout identity representation generalization”，就不构成 drift。

**Method Specificity: 8**

比上一版大幅提升。fusion function、fixed OOF reference bank、higher-is-better margin、query-side intervention、sample-key pairing、alpha 固定和 provenance hash 都足够具体。

剩余不够具体的点：

- `all-fit 同构路径 Delta` 的计算必须明确只作为 transfer audit，不能进入 training、threshold selection 或 fixed policy selection。
- Q1 replay 已经计算 mAP/Rank-1，但 hard gate 主要写 mean identity margin。最终目标是 mAP，建议把 OOF replay mAP non-inferiority 也列为 hard fail-fast gate。
- `aggregate gain >1e-4` 仍偏任意，且可能只是浮点级/样本噪声级 improvement。

**Contribution Quality: 8**

主贡献更干净。删除 Gram descriptor 后，贡献不再像“target 修复 + 表示不变性”双线并行，而是一个 paired-cache deployment-alignment 机制。这个方向比堆模块更优雅。

还没到 9 的原因是：它现在更像严谨工程化闭环和监督定义修复，novelty 需要最终结果和 post-success 两个 deletion checks 支撑，否则顶会 reviewer 可能认为是“正确 target engineering”。

**Frontier Leverage: 8**

foundation-model-era 使用合适：冻结 CLIP/Signal/V8 表征，把 OOF complete model 当 teacher，用小 Router 学控制策略。不强塞 DINO、LLM、extra backbone，这是优点。

**Feasibility: 8**

实现风险低于上一版。paired cache + replay evaluator 是可执行的，alpha 固定也降低自由度。主要风险是目标信号本身是否够强，以及 all-fit student feature 是否能泛化到 30-dev。

**Validation Focus: 7**

比上一版好很多：Q0 health、action-transfer、per-fold non-inferiority、aggregate nontrivial gain、OOF replay 都是正确的 fail-fast 结构，而且不等于主结果前消融。

扣分点是 gate 仍偏弱：

- `1e-6` 只是数值容差，不是科学 improvement。
- `>1e-4` 没绑定到 margin scale、oracle gap 或 mAP。
- replay mAP/Rank-1 目前像 report，不像 hard gate。

**Venue Readiness: 6**

低于 7。

Priority: IMPORTANT

弱点：目前仍是 proposal，没有 Q0/Q1/main dev 结果；而且即使 Q1 过门，single seed 30-dev 成功前也不能主张 deployable gain。顶会级 readiness 还依赖主结果和两项 post-success deletion check：`actual-path -> residual-only`、`paired deployment input -> fold raw input`。

修复：不扩大实验菜单。先跑固定 Q0/Q1；若 dev≥65，再只做这两项 claim-critical deletion checks。

### 重点问题检查

**1. all-fit student feature 做 identity-OOF Router validation 是否科学有效？**

有效，但 claim 边界必须窄。

它不是严格的 identity-OOF feature generalization，因为 all-fit V8 Phase-A backbone 已经见过全部 141 fit identities，包括 Q1 heldout fold identities。因此，如果说“Router 输入表征在 heldout identity 上未见过”，这是不准确的。

但它仍科学有效于当前目的：final inference 确实会读取这个 all-fit deployment feature；training target 来自 OOF teacher Delta；heldout evaluation 把 student weights 回放到 OOF teacher retrieval path。也就是说，它验证的是“deployment feature 上学到的 policy 是否能预测 identity-isolated OOF utility”，不是“feature extractor 是否未见 heldout identity”。

建议表述：

> Q1 is an identity-heldout policy validation under a fixed all-fit deployment feature interface, with labels and retrieval evaluation supplied only by OOF teacher paths. It does not claim identity-heldout representation training.

**2. OOF teacher action semantics 到 all-fit expert action 的 transfer gate 是否足够且不泄漏？**

基本足够，不构成标签泄漏，前提是 all-fit Delta 严格只用于 one-shot eligibility audit，不进入 Router loss、不调 threshold、不选 slot policy。

当前 transfer gate 检查“OOf oracle slot 在 all-fit Delta 上 aggregate 胜过 train-fold fixed slot、per-fold 不劣”。这是合理的语义转移最低门槛。它证明 slot 名称如 CNN/RGB、Mamba/TI 在 teacher 与 deployment expert 中没有完全错位。

但它只检查 oracle-selected action 的离散转移，不检查 soft target 分布整体转移。不是 blocking issue，但可以补一个只读 report：OOF Delta 与 all-fit Delta 的 sign agreement/top-k overlap 或 rank correlation。不要把它变成新贡献，也不要用它调参。

**3. fixed uniform gallery bank + query-side removal 与 final per-sample query/gallery routing 是否仍有 proxy mismatch？**

仍有 proxy mismatch，但已经不是 blocking。

Q0 target 是 query-side marginal utility against fixed uniform gallery bank；final dev 是 query/gallery 都由 Router per-sample 产生权重。因此 target 本身仍是训练代理，而不是 final pairwise retrieval objective 的完全展开梯度。

但 Q1 replay 正是用来关闭这个缺口：Router 产生 query/gallery weights 后，回放到 OOF baseline/residual，计算真实 filtering 下的 margin/mAP/Rank-1。如果 replay mAP/margin 不过门，就不 final dev。这个设计把 proxy mismatch 从“隐藏假设”降级成“被 gate 检查的风险”。

建议把 OOF replay mAP non-inferiority 明确列入 hard gate。

**4. Q1 OOF replay 是否真正闭合 student decision 与 teacher retrieval path？**

是的，机制上闭合得比 Round 0 好很多：

- student feature 只负责 Router 决策；
- decision 通过 sample key 映射到 teacher_oof_baseline/modal_residual；
- teacher path 重跑 exact `F_f`；
- same camera / cross-camera filtering 沿用 evaluator；
- bank 和 feature provenance hash 绑定。

这已经能回答“student policy 是否在 non-saturated OOF retrieval path 上真有收益”。

**5. 阈值是否过弱/任意？**

是。`1e-6` 是浮点容差，不能证明科学收益。`aggregate gain >1e-4` 也没有和 margin scale、slot oracle gap、query count 或 mAP 绑定。

更稳的最小修复：

- 保留 per-fold non-inferiority；
- aggregate gate 改成相对 attainable gap，例如 `learned - fixed > 0` 且达到 `oracle - fixed` 的预注册最小比例；
- 或者用 query-level paired bootstrap/paired sign test 证明 aggregate improvement 非偶然；
- replay mAP 至少不低于 fixed，aggregate replay mAP 必须严格高于 fixed。

这不增加实验，只加强同一次 Q1 的判定规则。

### Drift Warning

No major problem drift.

Minor claim-boundary warning: 使用 all-fit student feature 后，V13 不应声称“Router 输入路径对 heldout fit identities 完全未训练”。正确 claim 是“OOF teacher supervision and OOF replay are identity-heldout; student feature interface is the fixed final deployment interface.”

### Simplification Opportunities

1. 把 Q0 target generation、Q1 replay、final dev 都强制调用同一个 `F(x; w)` 实现，避免三个路径各自实现 fusion。
2. 把 all-fit Delta transfer audit 明确降级为 binary prerequisite，不作为 contribution 或 tunable metric。
3. 若 Q1 gate 已有 replay margin/mAP，Top-slot accuracy 可以保留为 Anchor-required diagnostic，不要扩大成额外故事线。

### Modernization Opportunities

NONE.

当前 frontier leverage 已合适：冻结 foundation representation + OOF utility distillation + small controller。不要加 DINO、HFER、LLM、extra backbone 或 MoE top-k。

### Remaining Action Items

Priority: CRITICAL

1. 明确写出 all-fit student feature 的科学边界：它见过 141 fit identities；Q1 验证的是 policy/target/replay，不是 representation OOF。
2. 把 all-fit Delta transfer audit 隔离为 read-only gate：不进入 training、不选 threshold、不选 fixed policy、不影响 loss。
3. 将 OOF replay mAP non-inferiority 和 aggregate replay mAP gain 加入 hard Q1 gate。

Priority: IMPORTANT

4. 替换或解释 `>1e-4` aggregate threshold，使其与 oracle gap、margin scale 或 paired-query significance 绑定。
5. 报告 OOF Delta vs all-fit Delta 的 sign/top-k/rank consistency，作为 action semantics sanity check。
6. 确保 final refit 使用的 all-fit Phase-A checkpoint SHA 与 paired cache 的 `student_*` checkpoint SHA 完全一致。

### Verdict

**REVISE**

这是一次实质性改进，不需要 RETHINK。主贡献更聚焦、更简单，Problem Anchor preserved，foundation-model-era 路线也合理。还不能 READY，因为 overall <9，且 all-fit student feature 的 identity-OOF claim 边界和 Q1 gate 强度需要再收紧。

</details>
