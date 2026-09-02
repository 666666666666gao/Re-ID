# V16 SATR Round 1 Review

<details open>
<summary>GPT-5.5 xhigh 原始评审</summary>

**V16 SATR Round 1 原始评审**

我完整读取了 V16 proposal，并对照了 RDPT 段和 V15 CRDE postmortem。结论：
**方向是对的，机制比 V15 更贴近失败证据，但当前还不能 READY**。主要
blocking issue 是 Q1 no-SATR comparator 不公平/自相矛盾；另外若干固定阈值和
mask 细节还没有到可执行级别。

| 维度 | 分数 | 评语 |
|---|---:|---|
| Problem Fidelity | 8.5 | 保持了核心 anchor：不回到 Router/hidden injection，而是把互补转成训练期关系修复。 |
| Method Specificity | 6.5 | SATR 数学主体清楚，但 fixed constants、receiver-mask detach、coverage gate、Q1 comparator 定义仍不够硬。 |
| Contribution Quality | 7.0 | 不是 RDPT 改名，但仍属于 relational/mutual KD 大类；新意必须压缩到 Signal-anchor + two-peer intersection + one-sided repair。 |
| Frontier Leverage | 7.5 | exact Signal/CLIP relation anchor 用得自然；不需要额外 VLM/LLM/DINO/RL。 |
| Feasibility | 7.5 | 无新 trainable module，B×B margin 可承受；公平 comparator 会增加 Q1 训练量，但仍在单 3090 预算内。 |
| Validation Focus | 5.5 | Q1 当前 comparator 设计是主要问题：SATR 端训练 20 epoch，却想用未额外训练 checkpoint 当 no-SATR comparator。 |
| Venue Readiness | 6.5 | 方法故事已经变干净，但 fair validation 和 novelty framing 未锁死前，顶会 ready 不成立。 |

**OVERALL weighted score: 7.1 / 10**

### 1. SATR 是否只是 RDPT / Similarity KD / RKD / PMKD 改名

不是简单改名，但 novelty 边界很窄。

SATR 明显不同于本地 RDPT：RDPT 是 URGC reliability gap 选择单 peer，传 KL +
role payload，并用 private hinge 防止同质化；SATR 是 cross-camera triplet margin
教学，要求两个 peer 的 conservative min-margin 同时超过 exact Signal 和
receiver，不传 logits、embedding、role payload，也没有 inference operation。

但 SATR 仍落在 relational KD / mutual KD / hard relation teaching 的大类里。可
辩护贡献不能写成“关系蒸馏”“互教”“hard sample KD”，只能写成更窄的机制
组合：**exact Signal relation anchor + two-peer intersection + stop-grad
min-margin repair + fixed private inference**。

### 2. 数学正确性检查

主体数学基本成立：

- `m_e(q,p,n)=s(q,p)-s(q,n)` 方向正确，更大代表更好；
- `t_e=min(m_j,m_k)` 是保守 two-peer lower bound；
- `t_e >= max(m0,m_e)+delta_r` 能表达两个 peer 同时优于 Signal 与 receiver；
- `L_repair` 是 one-sided gap filling，不强迫复制 peer 坐标；
- `L_protect` 只保护 fused 不跌破 Signal reliable relation，和 SATR 不必然矛盾。

但需要修订三个细节：

- mask 里的 `m_e` 必须明确是 `stopgrad(m_e)`，loss 里的 `m_e` 才是 live；
- `delta_r / gamma_p / epsilon_p / lambda_b / lambda_r / lambda_p` 必须给出固定值；
- “repair coverage 非零且非饱和”必须有可执行定义。

### 3. Q1 comparator 是否公平

这是当前最大 blocking issue。SATR 端继续训练20 epoch而 no-SATR 只是 frozen
checkpoint，会把继续 ID/triplet fine-tuning 混入增益。最小公平 comparator 是：
每个 V12 fold 从同一 checkpoint 启动，使用同一 source IDs、optimizer scope、
epoch、final-only selection、seed/sampler/schedule，只把
`lambda_r=lambda_p=0`。这会额外训练一套 no-SATR endpoint，但这是证明 SATR
机制的最低成本。

### 4. Two-peer intersection 稀疏/共误风险

风险真实存在，但不应引入 Router、EMA 或 single-peer fallback。把 coverage
定义写死；若为空或近饱和则 fail-closed。triplet 是 GT-defined positive/negative，
且要求 `t_e>0` 和超过 exact Signal，比普通 mutual KD 稳。

### 5. Fixed inference fusion 能否转成 fused gain

机制上可行但不保证。branch margin 修复可能被 normalization/Signal prefix
稀释，Q1/D1 fused gain gate 应捕获这一点。不建议加 Router、learned alpha 或
fusion head；应明确 `L_protect` 是防退化约束，不是 fused-gain 充分条件。

### 低于7分维度的具体修复

- **Method Specificity 6.5 / CRITICAL**：锁死系数、detach、coverage gate 和
  matched comparator。
- **Validation Focus 5.5 / CRITICAL**：Q1 必须训练同预算 no-SATR comparator。
- **Venue Readiness 6.5 / IMPORTANT**：把 novelty 收窄为 failure-driven
  mechanism；one-sided protect 降为 safety constraint。

### Simplification Opportunities

1. 把 one-sided anchor protection 从 supporting contribution 降级为 fused safety
   constraint。
2. 删除“为了不额外训练 comparator”的表述。
3. 把 SATR mask 写成单个可执行布尔公式。

### Modernization Opportunities

NONE。不要加 Router、EMA teacher、额外 backbone、DINO、文本、VLM 或大实验
菜单。

### Drift Warning

Problem drift：NONE。Validation drift：若保留 frozen no-SATR endpoint comparator，
claim 会漂移成继续训练优于未训练起点，必须修掉。

**Verdict: REVISE**

</details>

