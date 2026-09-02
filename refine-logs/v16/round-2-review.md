# V16 SATR Round 2 Review

<details open>
<summary>GPT-5.5 xhigh 原始评审</summary>

# V16 SATR Round 2 Re-evaluation 原始评审

我完整读取了 `round-1-refinement.md` 和 `threshold-freeze-readonly.md`。结论：
**SATR 已经从 Round 1 的不公平对照状态明显推进到可归因的 method plan，但仍
建议 REVISE 一轮**。现在最大问题不再是 comparator，而是 **coverage activity
gate 的定义会把“修复成功导致教学集合下降”误判为失败**。

| 维度 | 分数 | 评语 |
|---|---:|---|
| Problem Fidelity | 9.0 | Anchor 保持良好：不回 Router、不回 hidden injection、不扩大到额外 backbone 或 reranking。 |
| Method Specificity | 8.0 | 数学、mask、stop-grad、pair selection、训练端点边界都清楚很多；coverage gate 仍需修正。 |
| Contribution Quality | 8.1 | 已收敛为一个主机制；protect 降级为 safety constraint 是正确的。新颖性窄但可辩护。 |
| Frontier Leverage | 8.2 | exact Signal / CLIP relation anchor 使用自然，不需要再加 VLM/LLM/DINO/EMA teacher。 |
| Feasibility | 8.3 | 双端点 Q1 增加训练量但仍在单3090可承受范围内；无新 trainable module。 |
| Validation Focus | 6.8 | matched comparator 已修好，但 coverage gate 的 full-trajectory 判定会制造错误失败风险。 |
| Venue Readiness | 7.2 | 方法故事可投稿化，但还需要把 gate 定义锁到不会自毁的程度。 |

**OVERALL SCORE: 8.13 / 10**

### 重点核查

1. **Matched comparator**：现在可以归因。两个端点的 sampler index 和
   augmentation draw hash 应写入 receipt。
2. **Exact-Signal hard pair**：基本明确；应钉死 `c_p != c_q` 中 `c` 为
   RGBNT201 物理 camera ID，不是 modality ID；negative 不必强制跨 camera。
3. **Optimizer scope**：开放相同 role adapters/residual projections/heads合理；
   Signal、shared CLIP tail 和 exact prefix 冻结。
4. **Coverage gate**：optimizer-0 evidence 只能证明初始集合非空非饱和。完整
   20 epoch 平均下限会错误否定“receiver 已补足 margin、mask 因而下降”的成功
   模型。应改为 fixed-initial activity gate；训练过程 coverage 只记录。
5. **Fixed constants**：fit-only activity calibration 足够，但应公开是否尝试过
   其他候选值，避免隐藏 threshold scan。
6. **Q1 claim/gate**：除 coverage 判定位置外，主 claim 与 gate 基本一致。

### 低于7分维度

**Validation Focus 6.8 / IMPORTANT**：把 `[0.5%,25%]` gate 固定在 initial /
epoch0 deterministic batches；训练期 coverage 只记录，不作为 fail 条件。

### Simplification Opportunities

1. coverage gate 简化为 initial activity gate；
2. 不再解释 frozen comparator 历史；
3. protect 保持 safety constraint，不进入贡献列表。

### Modernization Opportunities

NONE。不要加 Router、EMA、额外 teacher/backbone、DINO 或 VLM。

### Drift Warning

NONE。

### Remaining Action Items

1. coverage gate 改为 fixed-initial / epoch0 activity gate；
2. receipt 记录 threshold-freeze probe 的完整候选/一次性状态；
3. M0 增加 paired endpoint draw hash；
4. 明确 physical camera mask 与无 positive 时不 fallback。

**Verdict: REVISE**

</details>

