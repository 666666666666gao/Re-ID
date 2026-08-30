# 实验计划

**问题**：RGB–NIR–TIR 多模态目标重识别中的异构架构协同、动态质量变化与模态缺失
**方法主张**：通过 CNN、Transformer、Mamba 三类互补归纳偏置的阶段性交互、反事实可靠性融合和角色保持互教，学习比单主干或晚期拼接更判别、更稳健的身份表示。
**日期**：2026-08-31
**工作名称**：TriFusion-ReID（论文名称待查新后冻结）
**实现底座**：DeMo，官方提交 `b4f323a430b32e3a1637c3e7acb25868cb52e9cd`，MIT License
**可测高指标基线**：MDReID，官方提交 `3525ac2da1a2a90a5a160c930fac674b4f226f6c`，官方 RGBNT201 checkpoint 报告 82.1 mAP / 85.2 Rank-1
**首要数据集**：RGBNT201，压缩包 SHA-256 `407f3b6d410cbf8e27127f9a0a6881ce3ba0006073845b48904a73e51479a94f`

**已完成前置门（本机证据）**：RGBNT201、MSVR310、RGBNT100 全量审计通过；Mamba 2.2.6.post3 / causal-conv1d 1.6.0 的 SM89 CUDA 前反向通过；MDReID 严格加载官方 checkpoint 后复现 82.0868 mAP / 85.1675 Rank-1，与公开值四舍五入一致。方法公式与接口冻结于 `docs/METHOD_SPEC_V1.md`。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1：异构三分支的深层协同，而非模型堆叠，能产生可测的互补身份信息 | 直接回应“收益只来自更多参数/集成”的审稿质疑 | 同预训练、同协议、参数/FLOPs 匹配；显著优于三个单分支、三分支平均/拼接和无交互版本；分支错误具有互补性，交互后每个分支自身也改善 | B2、B3 |
| C2：可靠性与互补性联合建模、角色保持互教能同时提高完整模态与缺失/退化模态鲁棒性 | 使方法不只刷完整三模态榜单，而具有部署价值 | 完整模态主表达到第一梯队；六种缺失组合和受控噪声下优于 DeMo/PDRNet 类强基线；路由与真实边际贡献相关；三种子统计稳定 | B2、B3、B4 |
| Anti-claim：提升不是来自更强预训练、更大模型、重排序或测试时训练 | 确保 SOTA 比较公平 | 冻结资源声明；matched-capacity control；不使用测试标签、额外文本、TTT 或重排序；单列使用额外资源的方法 | B1、B3、B5 |

## 三项核心创新（主源查新后冻结版）

1. **HFER — Heterogeneous Full-Expert Relay**：CNN、Transformer、Mamba 都接收全部 RGB/NIR/TIR 输入，三者是“架构专家”而非“模态专属分支”。在匹配深度进行尺度对齐、双向低秩 relay 交换，并保留私有残差通道与独立监督，从而把架构归纳偏置互补与模态互补解耦。
2. **URGC — Unified Reliability-Guided Collaboration**：学习经校准的 `r(sample, modality, expert)` 后验；同一后验同时控制跨专家消息带宽、最终融合权重和缺失/退化模态抑制。训练期以 leave-one-expert-out 边际贡献提供 stop-gradient 反事实监督，避免把普通 attention gate 重命名为可靠性。
3. **RDPT — Rejectable Directional Peer Teaching**：逐样本只允许可靠性显著更高的专家向较弱专家传递身份分布、度量关系和跨模态一致性；教师侧 stop-gradient，可靠性差距不足或各方均不可靠时拒绝蒸馏。角色专属辅助目标和去冗余约束防止三路坍缩成同一表示。

主源审计没有发现上述四个条件的精确组合，但每个子机制均有强先例。因此论文只主张这一窄边界的统一协作机制，不使用未经穷尽检索支持的“首个/首次”措辞；完整边界见 `docs/RESEARCH_AUDIT_2026-08-31.md`。

## Paper Storyline

- 主文必须证明：官方强基线可复现；三分支不是简单集成；三项机制分别有贡献；完整模态、缺失模态与效率形成统一收益。
- 附录支持：不同扫描顺序、router 温度、relay 宽度、更多噪声强度、可视化、失败样本和跨数据集迁移。
- 有意裁剪：大规模超参数网格、使用验证标签挑阈值、额外文本/SAM/TTT、与主张无关的弱基线堆表。

## Experiment Blocks

### Block 1：数据、指标与公开基线奇偶校验

- Claim tested：后续比较建立在正确数据协议和可信基线之上。
- Why this block exists：RGBNT201 官方实现的 query/gallery 都指向 `test`，必须确认相机过滤和 evaluator 与论文一致。
- Dataset / split / task：RGBNT201 `train_171`；官方 `test` 协议。
- Compared systems：MDReID 官方 RGBNT201 checkpoint；DeMo 官方配置在现代 CUDA/PyTorch 下的兼容训练与评估。
- Metrics：mAP、Rank-1/5/10；样本数、身份数、三模态配对；参数量、FLOPs。
- Setup details：CLIP ViT-B/16；50 epochs；Adam；LR 3.5e-4；AMP。DeMo 本机校准固定 B32/K4（每批 8 个身份）和 test batch 64，不使用梯度累积虚构 B64：batch-hard/身份采样损失只观察单个微批次，跨微批累积并不等价于官方身份组成。MDReID 官方 checkpoint 继续作为高指标数值锚点。
- Success criterion：数据审计全通过；evaluator 小样本手算回归通过；MDReID 官方 checkpoint 与报告数值差不超过 0.3 pp；DeMo 官方配置训练结果与 79.0/82.3 的报告值偏差若超过 0.5 pp，先诊断后再进入主模型结论。
- Failure interpretation：环境、预训练权重、数据版本、随机性或 evaluator 不同；不得把不一致基线当作弱对手。
- Table / figure target：实现细节与复现表。
- Priority：MUST-RUN。

### Block 2：TriFusion 主结果

- Claim tested：完整三模态下，异构协同模型优于 DeMo 和晚期融合。
- Why this block exists：这是核心 anchor result。
- Dataset / split / task：先 RGBNT201；通过门后扩展 MSVR310、RGBNT100。
- Compared systems：MDReID、DeMo；CNN-only；Transformer-only；Mamba-only；三分支平均；三分支 concat+MLP；完整 TriFusion。
- Metrics：mAP、Rank-1/5/10；三种子均值±标准差；参数量、训练/激活 FLOPs、吞吐、峰值显存。
- Setup details：相同输入尺寸、数据增强、预训练资源和 evaluator；不使用 reranking、TTT、文本或测试标签；主表固定种子 42/3407/9199。
- Success criterion：至少一个主数据集严格超过同资源、同协议赛道的公开最好结果；RGBNT201 静态 CLIP-B/16、256×128 赛道目标为同时超过 RoDI-CLIP 的 84.1 mAP / 87.2 Rank-1。RoDI-DINOv3 85.3/87.9、PMKD-DINOv2 84.7/88.9 和 ProxyTTT 85.0/88.5 分属更强预训练、多阶段蒸馏和测试时训练赛道，只作分轨上限，不能混合作公平 SOTA 声明。
- Failure interpretation：若仅 ensemble 有益而各分支不改善，C1 不成立；若只在一个小数据集涨，需限制泛化主张。
- Table / figure target：主表 Table 1、架构图 Figure 2。
- Priority：MUST-RUN。

### Block 3：三项创新隔离与容量控制

- Claim tested：HFER、URGC、RDPT 均有独立、机制一致的贡献。
- Compared systems：无协同；仅 HFER；HFER+普通 soft gate；HFER+URGC；HFER+URGC+无条件对称 KL；完整 RDPT；matched-parameter MLP/Transformer control；更多参数但无协同 control。
- Metrics：主指标、每分支独立 mAP/Rank-1、错误交并/Oracle routing 上限、router 边际贡献相关系数、表示 CKA/余弦冗余度。
- Setup details：先单种子决策，晋级配置再补三种子；任何选择仅用训练/验证预注册口径，不读取测试标签调参。
- Success criterion：每项核心机制相对直接前身带来可复现增益；完整方法优于容量匹配控制；三个分支均有非零梯度、非塌缩利用率和独立判别能力；可靠性在受控退化下具备可测校准性，并与真实边际贡献相关。
- Failure interpretation：无独立增益的模块从主创新中删除，不靠打包后的总增益保留。
- Table / figure target：Ablation Table 2；分支互补图 Figure 3。
- Priority：MUST-RUN。

### Block 4：缺失模态、质量退化与跨数据集

- Claim tested：URGC 与 RDPT 在真实不完整/低质量输入下有价值。
- Dataset / split / task：RGBNT201 六种固定缺失组合；模糊、遮挡、过曝/欠曝、噪声；MSVR310 完整与缺失协议；RGBNT100 在取得官方数据后加入。
- Compared systems：MDReID、DeMo、UGG/RoDI 式可靠性控制（能复现则优先复现）；完整 TriFusion；去掉 URGC/RDPT。
- Metrics：六组合平均 mAP/Rank-1、最差组合、退化曲线面积、完整模态性能保持率、ECE/路由校准。
- Setup details：退化强度在训练前冻结；报告每种模态和组合，不只报告平均值。
- Success criterion：完整模态不牺牲；缺失平均和最差组合均改善；router 权重随可控退化方向合理变化。
- Failure interpretation：若仅训练见过的 mask 有效，主张限于 modality-dropout 鲁棒性而非任意缺失泛化。
- Table / figure target：Robustness Table 3、退化曲线 Figure 4。
- Priority：MUST-RUN；跨数据集三种子为 NICE-TO-HAVE，至少单种子必须完成。

### Block 5：效率、可解释性与失败分析

- Claim tested：复杂度是可控且机制行为与设计一致。
- Compared systems：DeMo、dense 三分支、完整 TriFusion、可选 Top-2/Top-1 部署变体。
- Metrics：总/激活参数、FLOPs、images/s、峰值显存、分支权重分布、低质量样本案例、失败类型。
- Success criterion：完整模型在 8 GB 上可训练；部署变体给出明确性能—计算折中；不把只省理论 FLOPs 的 dense 实现称作稀疏。
- Failure interpretation：若三路始终全算，只声称动态融合，不声称稀疏推理。
- Table / figure target：Efficiency Table 4、定性 Figure 5。
- Priority：MUST-RUN。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | 数据/环境/指标 sanity | R000–R004 | 数据配对、CUDA、前向/反向、手算指标全通过 | 1–3 GPU-h | 旧代码与现代 PyTorch 不兼容 |
| M1 | 基线复现 | R010–R012 | MDReID checkpoint 差 ≤0.3 pp；DeMo 训练偏差受控，否则停止主模型训练并诊断 | 15–30 GPU-h | 8 GB 导致吞吐低；预训练权重入口受限 |
| M2 | 三个单分支与互补容量 | R020–R024 | 错误互补和 Oracle routing 上限足以支撑协同 | 25–50 GPU-h | 三分支高度同质 |
| M3 | 主方法单种子决策 | R030–R036 | 每项创新通过预注册增益/行为门 | 60–120 GPU-h | router 塌缩、蒸馏同质化、OOM |
| M4 | 正式结果 | R040–R049 | 主配置三种子、同协议 SOTA 与鲁棒性闭合 | 120–250 GPU-h | 本地算力周期长 |
| M5 | 论文证据完善 | R050–R059 | 效率、可视化、失败分析和结果审计完整 | 20–50 GPU-h | 只报最好点导致证据不完整 |

## Compute and Data Budget

- 总估计：首个可判定结果约 100–200 GPU-h；完整三数据集三种子约 250–500 GPU-h，实际以 M0/M1 吞吐实测更新。
- 本地硬件：RTX 4060 Laptop 8 GB；AMP 与保守 batch 为默认策略。Gradient accumulation 仅用于不破坏 batch-coupled 度量损失语义的配置；activation checkpointing、冻结/分阶段解冻按实测显存启用。
- 数据需求：RGBNT201、MSVR310 与 RGBNT100 均已取得并完成协议、配对和全量 JPEG 审计；下载来源、大小与 SHA-256 见 `../artifacts/download_manifest_20260831.json`。
- 人工评价：无。
- 最大瓶颈：8 GB 显存和本地训练时长，而非存储。

## Risks and Mitigations

- **风险：三个分支只是参数堆叠。** 缓解：参数/FLOPs 匹配控制、分支独立指标、错误互补与 leave-one-out 贡献。
- **风险：跨分支交互导致同质化。** 缓解：角色专属目标、relation-level 而非 raw feature 全对齐、去冗余约束。
- **风险：router 学成固定模态/固定分支映射。** 缓解：质量扰动、modality dropout、熵/负载审计、反事实边际监督。
- **风险：SOTA 使用更强 DINOv3 等预训练。** 缓解：分别报告同预训练公平表与开放资源上限表；必要时新增同一 DINOv3 tokenizer 的公平变体。
- **风险：MDReID 仓库未见明确开源许可证。** 缓解：仅把官方 checkpoint/代码作为不再分发的复现对照；新实现继续基于 MIT 许可的 DeMo，记录所有兼容补丁。
- **风险：单次随机种子偶然增益。** 缓解：决策阶段单种子，最终主结果固定三种子并报告方差。
- **风险：官方代码/README 不一致。** 缓解：固定提交、记录补丁、数据与权重哈希、先完成 baseline parity。

## Final Checklist

- [x] 主数据集原始压缩包和配对审计已记录
- [x] 主源新颖性、现代 RGBNT201 协议和分轨 SOTA 目标已审计
- [x] conda/CUDA/Mamba 环境已固化并完成真实前反向
- [x] MDReID 官方 checkpoint 已同协议复现并通过 parity 门
- [x] 三项创新的公式、软件契约、证伪门和消融矩阵已冻结
- [ ] 主论文表格覆盖完成
- [ ] 三项创新分别隔离
- [ ] 简洁性与容量匹配控制完成
- [ ] 不使用额外文本/TTT 的资源边界清楚
- [ ] NICE-TO-HAVE 与 MUST-RUN 已分离
- [ ] 三种子与结果哈希完成
- [ ] 同协议 SOTA 声明通过独立审计
