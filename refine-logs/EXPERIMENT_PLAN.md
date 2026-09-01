# TriFusion V5 baseline-preserving 主方法恢复实验计划

> 2026-09-01 19:00 更新：V4 已完成 60/60 epoch 并失败。最佳 fused
> `43.4031/42.7879`，低于 Mamba `44.0659/43.5152`，距 65 mAP dev 门
> `21.5969`。下方 V4 计划保留为历史预注册记录；当前唯一有效路线是先建立完整
> Signal baseline floor，再实现 baseline-preserving 三专家增量。此前“不复现
> baseline”的约束已被用户最新指令覆盖。

## 当前冻结决策

1. **先做完整 Signal baseline**：3072D=`3×512 direct CLIP global + 3×512 SIM`，包含 camera SIE；不得用 V4 的 1536D projected-CLS anchor 代称 Signal。
2. **同 checkpoint 双输出**：每个 V5 checkpoint 必须独立输出 `baseline_only` 和 `fused`；dev/result receipt 同时登记两者。
3. **梯度隔离保底**：先完成 baseline 阶段，再冻结 baseline 路径训练 CNN/Transformer/Mamba 增量；专家梯度不能改变 baseline-only 表征。
4. **晋级门**：同一 141-fit/30-dev 协议上，fused 必须不低于 baseline-only、必须高于每个专家并达到 65 mAP 才晋级；否则拒绝 fused，禁止融合增益主张。不增加运行时 fallback 分支。
5. **预算**：seed42 only；不做多种子；不做 baseline 矩阵/超参扫描；不做消融；dev 门通过前 official test access=0。

## V5 执行块

| Block | 内容 | 成功门 | 当前状态 |
|---|---|---|---|
| B0 | Signal commit/license/checkpoint/env/source parity 与 3072D feature contract | 远端环境可重建；完整 feature/SIE 可审计；upstream 与 local 标签分开 | COMPLETE—PASS |
| B1 | 同一 141/30 dev 协议训练/评估 baseline-only | 产生真实 baseline dev mAP/R1 与固定 checkpoint；无 official test | RUNNING |
| B2 | V5 TDD、容量与固定批门 | baseline bit/distance stable；专家梯度隔离；B32/K4 适配 3090 | PENDING |
| B3 | 一次 V5 60-epoch dev 主实验 | fused≥baseline、fused>所有专家、fused≥65 mAP | BLOCKED BY B0-B2 |
| B4 | 全171固定训练与 official once | 仅 B3 全门通过；正式 >85.3/>87.9 | BLOCKED |
| B5 | 消融 | 仅 B4 超目标后 | BLOCKED |

**问题**：V3 的 CNN、Transformer、Mamba 残差已经具备差异性和身份信息，但范数硬帽与近均匀路由使其在最终检索距离中的平方能量仅约 4.5%，融合无法超过最佳分支。
**方法主张**：在不丢失直接 CLIP 身份锚点的前提下，以非破坏式残差银行保存三个专家的独立信息，并用身份效用监督和无自由尺度的能量校准使专家信息真正影响检索距离。
**日期**：2026-09-01

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：非破坏式、能量平衡的三专家残差银行能把已有专家差异转化为融合增益 | V3 已证明残差不完全塌缩，但破坏式求和和约 0.216 的残差/锚点范数比使其影响过弱 | 固定 held-out dev 上 fused mAP≥65，且 fused 同时高于 direct anchor 与 CNN/Transformer/Mamba 最佳分支 | B1、B2 |
| C2：身份效用监督的路由能摆脱近均匀分配，并按样本/模态保留有效专家信息 | V3 路由归一化熵约 0.9998，质量路由名义存在但没有形成选择性 | B2 通过主门；正式结果超过 85.3/87.9 后，才允许以预注册消融验证路由与残差银行的独立贡献 | B2、条件 B4 |
| Anti-claim：收益只来自扩大维度或任意调大残差 | 需要排除“更多维度自然更好”的解释 | 仅在正式目标超过后进行等维、等能量控制；在此之前不消耗消融预算 | 条件 B4 |

## 冻结的单一结构修正

V4 保留 V3 的共享 12 层 CLIP、直接三模态 projected-CLS anchor、CNN/Transformer/Mamba 专家以及三阶段 HFER，只替换最终残差使用方式：

1. **非破坏式三专家残差银行**：不再沿专家维求和。分别保留 CNN、Transformer、Mamba 的三模态残差块，避免异构信息在求和时相互抵消。
2. **无自由尺度的距离能量校准**：先形成路由加权残差银行，再把整个残差银行的 L2 能量归一到 direct anchor 的 L2 能量；最终表示为 `[anchor, calibrated residual bank]`。不扫描残差倍率。
3. **身份效用路由监督**：在 B32/K4 身份批次内，以 detached 的逐样本 batch-hard 正负间隔构造三个专家的效用分布，监督路由预测；fused ID/triplet 仍端到端反传。该监督只使用训练身份和当前训练批次，不读 dev/test 标签。

这是一项统一的“energy-calibrated utility-routed residual bank”结构修正，不是学习率、温度、batch 或残差倍率扫描。

## Paper Storyline

- 主文必须证明：直接强锚点没有被覆盖；三专家残差被完整保留；融合在同一 held-out dev 上超过锚点和每个分支；最终同协议指标确实超过冻结目标后才谈 SOTA。
- 附录可以支持：达到正式目标后再做等维控制、去效用监督、去 HFER、单专家/双专家删除实验。
- 当前明确裁掉：baseline 复现矩阵、多种子、提前消融、官方 test 调参、残差倍率/温度/学习率网格搜索。单一 Signal baseline floor 已由用户明确授权且必须优先完成。

## Experiment Blocks

### Block 1：V4 工程和学习能力门禁

- Claim tested：V4 结构真实可训练，三个专家和路由均收到有限梯度，能量校准不产生 NaN/OOM。
- Dataset / split / task：RGBNT201 141-fit，仅训练批次；不迭代 dev/test loader。
- Compared systems：仅 V4 单一冻结结构。
- Metrics：参数量、峰值显存、AMP overflow、有限梯度覆盖、100-step 固定批 loss ratio、官方访问计数。
- Setup：seed42，B32/K4，RTX3090，沿用 V3 CLIP 与优化器配置；先专项 TDD，再 8-step capacity 和 100-step overfit。
- Success criterion：参数≤120M；峰值显存适配 24GB；梯度覆盖 100%；无非有限值/overflow；固定批 loss ratio≤0.10；official access=0。
- Failure interpretation：结构或训练接口不具备主实验资格，停止在工程层修复，不启动 60 epoch。
- Table / figure target：补充材料工程审计表。
- Priority：MUST-RUN。

### Block 2：V4 完整 held-out dev 主门

- Claim tested：能量平衡的非破坏式路由银行把专家差异转化为融合增益。
- Dataset / split / task：固定 141-fit/30-dev，825 query/825 gallery，无 official test。
- Compared systems：同一 V4 checkpoint 的 fused、direct anchor、CNN、Transformer、Mamba 输出；不是消融。
- Metrics：mAP、Rank-1/5/10；路由熵；专家/锚点能量比；fused 与最佳分支差值。
- Setup：seed42 only，B32/K4，60 epoch，固定 config，按 fused dev mAP 选 checkpoint；不热改。
- Success criterion：fused mAP≥65，且 fused mAP 同时高于 anchor 和所有专家；无 fatal/nonfinite；official access=0。
- Failure interpretation：C1/C2 不成立；记录负结果，继续主方法结构恢复，但仍不做消融或官方评估。
- Table / figure target：主结果晋级表与路由诊断图。
- Priority：MUST-RUN。

### Block 3：条件式 postfreeze-final

- Claim tested：通过 dev 门的冻结 V4 能在全部 171 训练身份上迁移到官方 RGBNT201 test。
- Dataset / split / task：all-171 训练；固定终点后官方 test 只评估一次。
- Compared systems：V4 fused、anchor、三分支；公开方法仅作严格标注的同协议参照。
- Metrics：mAP、Rank-1/5/10。
- Setup：只有 B2 全部门通过才生成正式身份；一次 official access，不再选模。
- Success criterion：严格超过 85.3 mAP 和 87.9 Rank-1，且 fused 高于最佳分支。
- Failure interpretation：不支持 SOTA；不进入消融。
- Table / figure target：论文主表。
- Priority：CONDITIONAL MUST-RUN。

### Block 4：条件式主张消融

- 仅当 Block 3 严格超过冻结目标后解锁。
- 预注册候选：去残差银行、去身份效用监督、去 HFER、等维/等能量控制、CNN/Transformer/Mamba 单删和双分支控制。
- 在解锁前不实现、不排队、不运行。
- Priority：BLOCKED-CONDITIONAL。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | V3 失败闭环 | V3 60-epoch 终态、冻结最佳分解、独立 result-to-claim | 已完成；claim_supported=no | 已消耗 | 仅相关性诊断，不能当因果救援 |
| M1 | V4 TDD 与真实 3090 门禁 | V4-R000/001/002 | 全部工程门通过才允许主训 | 约 15–25 分钟 | 6144 维 fused head 增加显存；120M 参数上限 fail-closed |
| M2 | V4 完整 dev 主实验 | V4-R003 | fused≥65 且高于 anchor/最佳分支 | 约 45–60 分钟 | identity-utility target 噪声；固定结构失败即记录，不扫参 |
| M3 | 一次正式评估 | V4-R004 | dev 全门通过后才运行；正式 >85.3/>87.9 | 约 50–70 分钟 | 官方预算一次，启动前冻结全部哈希 |
| M4 | 论文消融 | V4-R005+ | 仅正式目标超过后解锁 | 未估算 | 当前禁止 |

## Compute and Data Budget

- GPU：仅远端单张 RTX3090 24GB；本地 Windows/WSL 只传输代码与文档。
- Seeds：仅 42。
- 数据：RGBNT201 现有审计副本；不下载、不复制到本地。
- 首个可执行预算：V4 工程门约 0.4 GPU-hour，完整 dev 约 1 GPU-hour。
- 最大瓶颈：不是显存，而是 141 身份上的身份外泛化与路由效用估计。

## Risks and Mitigations

- 风险：等能量残差会放大噪声。缓解：只允许身份效用路由选择，不引入可扫描倍率；仍以 fused>anchor/best branch 作为硬门。
- 风险：路由重新退化为均匀。缓解：记录归一化熵与训练效用目标一致性；均匀且不增益即判失败。
- 风险：高维表示靠维度而非机制。缓解：正式目标通过后才运行等维/等能量控制，当前不以维度解释论文主张。
- 风险：dev 波动。缓解：冻结 fused-mAP 选点、60 个 epoch 全记录，并对最佳 checkpoint 确定性复评；不多种子、不平滑选点。

## Final Checklist

- [x] V3 60-epoch 负结果与 checkpoint 分解完成
- [x] 官方访问保持 0，V3 不晋级
- [x] V4 单一结构修正和成功/失败边界冻结
- [x] V4 TDD、capacity、overfit 门通过（95,197,266 参数；峰值 6,548 MiB reserved；梯度覆盖 100%；固定批 loss ratio 0.06677；official0）
- [x] V4 完整 dev 主门完成但失败（epoch27 fused `43.4031/42.7879`；低于 Mamba `0.6628 mAP`；official0）
- [ ] 仅满足 dev 门后执行正式一次评估
- [ ] 仅正式目标超过后解锁消融与 SOTA 对比
- [ ] Signal 完整 3072D baseline-only 路径与同协议 dev 指标
- [ ] V5 baseline-only/fused 双输出、梯度隔离和 fused 晋级门禁
